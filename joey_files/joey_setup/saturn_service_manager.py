"""
SATURN Service Manager for AI Service Discovery.

This module provides zero-configuration AI service discovery using DNS-SD
(DNS Service Discovery). It discovers SATURN-compatible services on the local
network and manages service connections with automatic failover and health monitoring.

Based on the SATURN protocol using DNS-SD subprocess commands for service discovery.
"""

import subprocess
import socket
import time
import requests
import threading
import re
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SaturnService:
    """Represents a discovered SATURN service."""
    name: str
    url: str
    priority: int
    ip: str
    features: str = ""
    api: str = ""
    version: str = ""
    last_seen: datetime = None

    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()


class ServiceDiscovery:
    """Background service discovery using DNS-SD subprocess commands."""

    def __init__(self, discovery_interval: int = 10, on_service_change: Optional[Callable] = None):
        """
        Initialize background service discovery.

        Args:
            discovery_interval: Seconds between discovery scans
            on_service_change: Callback function(action, name, url, priority) for service changes
        """
        self.services: Dict[str, SaturnService] = {}
        self.lock = threading.Lock()
        self.running = True
        self.discovery_interval = discovery_interval
        self.on_service_change = on_service_change
        self.thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.thread.start()

    def _discovery_loop(self):
        """Continuously discover services in background."""
        while self.running:
            try:
                self._discover_services()
            except Exception as e:
                print(f"[SATURN] Discovery error: {e}")
            time.sleep(self.discovery_interval)

    def _discover_services(self):
        """Single discovery pass using DNS-SD."""
        discovered = self._run_dns_sd_discovery()
        if discovered is None:
            return

        current_time = datetime.now()
        discovered_names = set()

        with self.lock:
            # Update or add discovered services
            for svc in discovered:
                discovered_names.add(svc['name'])

                if svc['name'] not in self.services:
                    # New service
                    self.services[svc['name']] = SaturnService(
                        name=svc['name'],
                        url=svc['url'],
                        priority=svc['priority'],
                        ip=svc['ip'],
                        features=svc.get('features', ''),
                        api=svc.get('api', ''),
                        version=svc.get('version', ''),
                        last_seen=current_time
                    )
                    print(f"[SATURN] Discovered service: {svc.get('api', 'Unknown')} at {svc['url']} (priority {svc['priority']})")
                    if self.on_service_change:
                        self.on_service_change('added', svc['name'], svc['url'], svc['priority'])
                else:
                    # Update existing
                    service = self.services[svc['name']]
                    service.url = svc['url']
                    service.priority = svc['priority']
                    service.ip = svc['ip']
                    service.features = svc.get('features', '')
                    service.api = svc.get('api', '')
                    service.version = svc.get('version', '')
                    service.last_seen = current_time

            # Remove services that disappeared
            removed = [name for name in self.services.keys() if name not in discovered_names]
            for name in removed:
                service = self.services[name]
                print(f"[SATURN] Service removed: {service.url}")
                del self.services[name]
                if self.on_service_change:
                    self.on_service_change('removed', name, service.url, service.priority)

    def _run_dns_sd_discovery(self) -> Optional[List[dict]]:
        """Run dns-sd discovery and return list of services."""
        services = []

        try:
            # Browse for services
            browse_proc = subprocess.Popen(
                ['dns-sd', '-B', '_saturn._tcp', 'local'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(2.0)
            browse_proc.terminate()

            try:
                stdout, stderr = browse_proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                browse_proc.kill()
                stdout, stderr = browse_proc.communicate()

            # Parse service names
            service_names = []
            for line in stdout.split('\n'):
                if 'Add' in line and '_saturn._tcp' in line:
                    parts = line.split()
                    if len(parts) > 6:
                        service_names.append(parts[6])

            # Get details for each service
            for service_name in service_names:
                try:
                    lookup_proc = subprocess.Popen(
                        ['dns-sd', '-L', service_name, '_saturn._tcp', 'local'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    time.sleep(1.5)
                    lookup_proc.terminate()

                    try:
                        stdout, stderr = lookup_proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        lookup_proc.kill()
                        stdout, stderr = lookup_proc.communicate()

                    hostname = None
                    port = None
                    priority = 50
                    features = ""
                    api = ""
                    version = ""

                    for line in stdout.split('\n'):
                        if 'can be reached at' in line:
                            match = re.search(r'can be reached at (.+):(\d+)', line)
                            if match:
                                hostname = match.group(1).rstrip('.')
                                port = int(match.group(2))

                        if 'priority=' in line:
                            parts = line.split('priority=')
                            if len(parts) > 1:
                                priority_str = parts[1].split()[0]
                                priority = int(priority_str)

                        if 'features=' in line:
                            parts = line.split('features=')
                            if len(parts) > 1:
                                features = parts[1].split()[0]

                        if 'api=' in line:
                            parts = line.split('api=')
                            if len(parts) > 1:
                                api = parts[1].split()[0]

                        if 'version=' in line:
                            parts = line.split('version=')
                            if len(parts) > 1:
                                version = parts[1].split()[0]

                    if hostname and port:
                        try:
                            ip_address = socket.gethostbyname(hostname)
                        except socket.gaierror:
                            ip_address = hostname

                        service_url = f"http://{ip_address}:{port}"
                        services.append({
                            'name': service_name,
                            'url': service_url,
                            'priority': priority,
                            'ip': ip_address,
                            'features': features,
                            'api': api,
                            'version': version
                        })

                except (subprocess.TimeoutExpired, ValueError, IndexError):
                    continue

        except FileNotFoundError:
            print("[SATURN] ERROR: dns-sd not found. Please install Bonjour services (Windows) or ensure dns-sd is available.")
            return None
        except Exception as e:
            print(f"[SATURN] Discovery error: {e}")
            return None

        # Deduplicate by name, preferring non-loopback
        unique_services = {}
        for svc in services:
            name = svc['name']
            ip = svc['ip']
            is_loopback = ip.startswith('127.') or ip == 'localhost'

            if name not in unique_services:
                unique_services[name] = svc
            else:
                existing = unique_services[name]
                existing_is_loopback = existing['ip'].startswith('127.') or existing['ip'] == 'localhost'

                if (svc['priority'] < existing['priority']) or \
                   (svc['priority'] == existing['priority'] and existing_is_loopback and not is_loopback):
                    unique_services[name] = svc

        return list(unique_services.values())

    def get_all_services(self) -> List[SaturnService]:
        """Get all discovered services sorted by priority."""
        with self.lock:
            services = list(self.services.values())
            return sorted(services, key=lambda s: s.priority)

    def get_best_service(self) -> Optional[SaturnService]:
        """Get service with lowest priority (highest preference)."""
        with self.lock:
            if not self.services:
                return None
            return min(self.services.values(), key=lambda s: s.priority)

    def stop(self):
        """Stop background discovery."""
        self.running = False


class SaturnServiceManager:
    """
    Manages SATURN service discovery and connection.

    Automatically discovers AI services on the local network using DNS-SD
    and provides methods to interact with them via OpenAI-compatible API.
    """

    def __init__(self, discovery_timeout: float = 3.0):
        """
        Initialize the service manager.

        Args:
            discovery_timeout: How long to wait for initial service discovery (seconds)
        """
        print(f"[SATURN] Searching for services...")

        # Start background discovery
        self.discovery = ServiceDiscovery(discovery_interval=10)

        # Wait for initial discovery
        time.sleep(discovery_timeout)

        services = self.discovery.get_all_services()
        if not services:
            print(f"[SATURN] Warning: No services found within {discovery_timeout}s timeout")
        else:
            print(f"[SATURN] Found {len(services)} service(s)")
            for service in services:
                print(f"[SATURN]   - {service.api} at {service.url} (priority {service.priority})")

    def get_best_service(self) -> Optional[SaturnService]:
        """Get the highest-priority (lowest priority number) service."""
        return self.discovery.get_best_service()

    def get_all_services(self) -> List[SaturnService]:
        """Get all discovered services sorted by priority."""
        return self.discovery.get_all_services()

    def check_health(self, service: SaturnService) -> bool:
        """
        Check if a service is healthy.

        Args:
            service: The service to check

        Returns:
            True if service responds to health check, False otherwise
        """
        try:
            response = requests.get(f"{service.url}/v1/health", timeout=5)
            return response.ok
        except Exception as e:
            print(f"[SATURN] Health check failed for {service.url}: {e}")
            return False

    def get_models(self, service: SaturnService) -> List[dict]:
        """
        Get available models from a service.

        Args:
            service: The service to query

        Returns:
            List of model dictionaries
        """
        try:
            response = requests.get(f"{service.url}/v1/models", timeout=10)
            if response.ok:
                data = response.json()
                return data.get('models', [])
            return []
        except Exception as e:
            print(f"[SATURN] Failed to get models from {service.url}: {e}")
            return []

    def chat_completion(self, messages: List[dict], model: Optional[str] = None,
                       max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                       timeout: int = 120) -> Optional[dict]:
        """
        Make a chat completion request to the best available service.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (if None, uses first available model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (ignored by some models)
            timeout: Request timeout in seconds

        Returns:
            Response dict in OpenAI format, or None if all services fail
        """
        services = self.discovery.get_all_services()

        if not services:
            raise Exception("No SATURN services available. Please start a SATURN server.")

        # Try services in priority order
        for service in services:
            try:
                # Get available models if model not specified
                if model is None:
                    models = self.get_models(service)
                    if not models:
                        print(f"[SATURN] No models available from {service.url}, trying next service")
                        continue
                    model = models[0]['id']
                    print(f"[SATURN] Using model: {model}")

                # Build request payload
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False
                }

                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens

                if temperature is not None:
                    payload["temperature"] = temperature

                # Make request
                print(f"[SATURN] Requesting from {service.url} (priority {service.priority})")
                response = requests.post(
                    f"{service.url}/v1/chat/completions",
                    json=payload,
                    timeout=timeout
                )

                if response.ok:
                    return response.json()
                else:
                    print(f"[SATURN] Request failed with status {response.status_code}: {response.text[:200]}")

            except Exception as e:
                print(f"[SATURN] Error with service {service.url}: {e}")
                continue

        # All services failed
        print("[SATURN] All services failed")
        return None

    def close(self):
        """Clean up resources."""
        self.discovery.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
