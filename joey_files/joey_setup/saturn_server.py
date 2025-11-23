"""SATURN server for AI service discovery.

Lightweight OpenRouter proxy with full model catalog support.
Implements SATURN service discovery protocol using DNS-SD.

Configuration:
- OPENROUTER_API_KEY: Set in .env file
- OPENROUTER_BASE_URL: https://openrouter.ai/api/v1/chat/completions (default)
- Port: Auto-selected or specify with --port
- Priority: Default 50, lower values = higher priority

Usage:
    python joey_setup/saturn_server.py
    python joey_setup/saturn_server.py --port 8080 --priority 10
"""

import argparse
import socket
import os
import json
import time
import subprocess
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import requests
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "Missing OPENROUTER_API_KEY. "
        "Please set OPENROUTER_API_KEY in your .env file"
    )


class ModelCache:
    """Cache for OpenRouter models with automatic refresh."""
    def __init__(self):
        self.models: List[Dict[str, Any]] = []
        self.last_updated: Optional[datetime] = None
        self.lock = threading.Lock()

    def update(self, models: List[Dict[str, Any]]):
        with self.lock:
            self.models = models
            self.last_updated = datetime.now()

    def get(self) -> List[Dict[str, Any]]:
        with self.lock:
            return self.models.copy()

    def needs_refresh(self, max_age_hours: int = 1) -> bool:
        with self.lock:
            if not self.last_updated:
                return True
            return datetime.now() - self.last_updated > timedelta(hours=max_age_hours)


model_cache = ModelCache()


def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch available models from OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }

    try:
        print("[Server] Fetching models from OpenRouter API...")
        response = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = data.get("data", data)

        formatted_models = []

        auto_model = {
            "id": "openrouter/auto",
            "object": "model",
            "owned_by": "openrouter",
            "context_length": None,
            "pricing": None,
            "modality": "multimodal",
            "description": "Intelligent routing to best model via NotDiamond"
        }
        formatted_models.append(auto_model)

        for model in models:
            if isinstance(model, dict) and "id" in model:
                formatted_models.append({
                    "id": model["id"],
                    "object": "model",
                    "owned_by": model.get("owned_by", "openrouter"),
                    "context_length": model.get("context_length"),
                    "pricing": model.get("pricing"),
                    "modality": model.get("modality", "text"),
                })

        print(f"[Server] Successfully fetched {len(formatted_models)} models from OpenRouter")
        return formatted_models
    except requests.RequestException as e:
        print(f"[Server] Failed to fetch OpenRouter models: {e}")
        return []


async def refresh_models_if_needed():
    """Refresh model cache if it's stale."""
    if model_cache.needs_refresh():
        print("[Server] Model cache is stale, refreshing...")
        models = fetch_openrouter_models()
        if models:
            model_cache.update(models)
            print(f"[Server] Successfully refreshed cache with {len(models)} models")
        else:
            print("[Server] Failed to refresh models, keeping existing cache")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup for the FastAPI app."""
    print("=" * 60)
    print("[Server] Starting up...")
    print("=" * 60)

    models = fetch_openrouter_models()
    if models:
        model_cache.update(models)
        print(f"[Server] Cached {len(models)} models from OpenRouter")
    else:
        print("[Server] WARNING: Failed to fetch models at startup. The /v1/models endpoint will be empty.")

    yield

    print("[Server] Shutting down...")


app = FastAPI(
    title="Saturn Server",
    description="SATURN server with full OpenRouter model catalog",
    version="2.0",
    lifespan=lifespan
)


class UserAIRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    stream: bool = False


@app.get("/v1/health")
async def health() -> dict:
    """Return server health status."""
    return {
        "status": "ok",
        "provider": "OpenRouter",
        "models_cached": len(model_cache.get()),
        "features": ["full-catalog", "auto-routing", "streaming"]
    }


@app.get("/v1/models")
async def get_models() -> dict:
    """Return list of available models from OpenRouter API."""
    await refresh_models_if_needed()

    cached_models = model_cache.get()

    if not cached_models:
        raise HTTPException(
            status_code=503,
            detail="No models available. Failed to fetch from OpenRouter API."
        )

    return {"models": cached_models}


@app.post("/v1/chat/completions")
async def chat_completions(request: UserAIRequest):
    """Handle chat completion requests by proxying to OpenRouter."""
    print(f"Received request for model: {request.model}")
    print(f"Messages count: {len(request.messages)}, stream: {request.stream}")

    openrouter_request = {
        "model": request.model,
        "messages": request.messages
    }
    if request.max_tokens is not None:
        openrouter_request["max_tokens"] = request.max_tokens
    if request.stream:
        openrouter_request["stream"] = True

    print(f"Forwarding to OpenRouter with model: {request.model}, stream: {request.stream}")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=openrouter_request,
            timeout=30,
            stream=request.stream
        )

        print(f"OpenRouter response status: {response.status_code}")

        if not response.ok:
            print(f"OpenRouter error response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRouter API error: {response.text}"
            )

        if request.stream:
            print("Returning streaming response")

            def generate():
                try:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                data_content = decoded_line[6:]

                                if data_content == '[DONE]':
                                    yield f"data: [DONE]\n\n".encode('utf-8')
                                    break

                                try:
                                    chunk_data = json.loads(data_content)
                                    yield f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                                except json.JSONDecodeError:
                                    continue
                finally:
                    response.close()

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            try:
                result = response.json()
                print("OpenRouter response parsed successfully")
                return result
            except requests.exceptions.JSONDecodeError:
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenRouter returned non-JSON response. Status: {response.status_code}"
                )

    except requests.Timeout:
        print("OpenRouter request timed out")
        raise HTTPException(status_code=504, detail="OpenRouter request timed out")
    except requests.RequestException as e:
        print(f"OpenRouter connection error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OpenRouter connection error: {str(e)}")


def find_available_priority(desired_priority: int, service_type: str) -> int:
    """Scan for existing services and avoid priority collisions using DNS-SD."""
    priorities = set()

    try:
        browse_proc = subprocess.Popen(
            ['dns-sd', '-B', '_saturn._tcp', 'local'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(2.0)
        browse_proc.terminate()

        stdout, _ = browse_proc.communicate(timeout=1)

        for line in stdout.split('\n'):
            if '_saturn._tcp' in line:
                try:
                    service_name = line.split()[6] if len(line.split()) > 6 else None
                    if service_name:
                        lookup_proc = subprocess.run(
                            ['dns-sd', '-L', service_name, '_saturn._tcp'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=2
                        )
                        for lookup_line in lookup_proc.stdout.split('\n'):
                            if 'priority=' in lookup_line:
                                parts = lookup_line.split('priority=')
                                if len(parts) > 1:
                                    priority_str = parts[1].split()[0]
                                    priorities.add(int(priority_str))
                except (IndexError, ValueError, subprocess.TimeoutExpired):
                    continue
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("dns-sd not available or timed out, using desired priority without checking")
        return desired_priority

    current_priority = desired_priority
    while current_priority in priorities:
        print(f"Priority {current_priority} is already in use, trying {current_priority + 1}...")
        current_priority += 1

    if current_priority != desired_priority:
        print(f"Adjusted priority from {desired_priority} to {current_priority}")

    return current_priority


def register_saturn(port: int, priority: int, service_type: str) -> subprocess.Popen:
    """Register this server as a SATURN service via DNS-SD."""
    actual_priority = find_available_priority(priority, service_type)

    host = socket.gethostname()
    service_name = f"SaturnServer"

    txt_records = f"version=2.0 api=OpenRouter features=full-catalog,auto-routing,streaming priority={actual_priority}"

    try:
        registration_proc = subprocess.Popen(
            [
                'dns-sd', '-R',
                service_name, '_saturn._tcp', 'local',
                str(port), txt_records
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"{service_name} registered via dns-sd with priority {actual_priority}")
        return registration_proc
    except FileNotFoundError:
        print("ERROR: dns-sd not found. Please install Bonjour services (Windows) or ensure dns-sd is available.")
        return None


def find_port_number(host: str, start_port=8080, max_attempts=20) -> int:
    """Find an available port automatically."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports in range {start_port} - {start_port + max_attempts}")


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Saturn Server - OpenRouter Proxy")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (auto-select if not specified)")
    parser.add_argument("--priority", type=int, default=50, help="SATURN service priority")
    args = parser.parse_args()

    port = args.port if args.port else find_port_number(args.host)

    print("=" * 60)
    print("SATURN Server - OpenRouter Proxy with Full Model Catalog")
    print("=" * 60)
    print(f"Starting on {args.host}:{port} with desired priority {args.priority}")
    print(f"Models: Full OpenRouter catalog (343+ models including openrouter/auto)")
    print(f"Features: full-catalog, auto-routing, streaming, SATURN discovery")
    print("=" * 60)

    service_type = "_saturn._tcp.local."
    registration_proc = register_saturn(port, priority=args.priority, service_type=service_type)

    try:
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    finally:
        if registration_proc:
            print("\nUnregistering service...")
            registration_proc.terminate()
            registration_proc.wait(timeout=2)


if __name__ == "__main__":
    main()
