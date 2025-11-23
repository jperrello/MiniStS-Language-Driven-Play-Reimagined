# SATURN: Zero-Configuration AI Service Discovery

**SATURN eliminates API key management and hardcoded endpoints.** Agents automatically discover AI services on your local network via DNS-SD (DNS Service Discovery), just like printers and AirPlay devices.

## Setup

1. **Add your API key to `.env`** (one-time setup):
   ```bash
   echo "OPENROUTER_API_KEY=your-key-here" >> .env
   echo "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions" >> .env
   ```

2. **Start the SATURN test server**:
   ```bash
   python testing/saturn_test_server.py
   ```

   The server starts and announces itself via DNS-SD using the `_saturn._tcp.local.` service type.

3. **Run any SATURN-enabled agent**:
   ```bash
   python run_rcot_game.py
   ```

   Agents automatically discover and connect to the test server. No configuration needed.

## Running Tests

**Test the server**:
```bash
python testing/test_saturn_server.py
```

**Test agent gameplay** (agent vs agent):
```bash
python run_rcot_game.py
```

## How It Works

**Service Discovery**: Saturn uses DNS-SD (Multicast DNS Service Discovery) for zero-configuration networking
- **Service Type**: `_saturn._tcp.local.`
- **Discovery Method**: DNS-SD subprocess commands (`dns-sd -B` for browsing, `dns-sd -L` for lookup)
- **Registration**: Servers use `dns-sd -R` to register themselves on the network

**Architecture**:
- Agents use `SaturnServiceManager` to discover services via DNS-SD subprocess calls
- The manager continuously monitors for services in the background
- Automatic connection to the highest-priority service (lowest priority number)
- Automatic failover when services go offline
- No API keys in agent code—the server handles authentication

**Priority System**:
- Lower numbers = higher priority
- Default priority: 50
- Automatic priority conflict resolution
- Clients select the best service based on priority

**Result:** Simpler code, faster iteration, and zero configuration for agents.
