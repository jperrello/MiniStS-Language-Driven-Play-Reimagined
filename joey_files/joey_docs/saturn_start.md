# SATURN Test Server - Quick Start Guide

## 30-Second Setup

```bash
# 1. Add API key to .env file (in project root)
echo "OPENROUTER_API_KEY=your-key-here" >> .env
echo "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions" >> .env

# 2. Start the test server
python testing/saturn_test_server.py

# 3. Test it (in another terminal)
python testing/test_saturn_server.py
```

## Get an API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Go to [API Keys](https://openrouter.ai/keys)
4. Create a new key
5. Add it to your `.env` file

## Common Commands

```bash
# Start with default settings (port auto-selected, priority 50)
python testing/saturn_test_server.py

# Start with high priority (will be preferred by clients)
python testing/saturn_test_server.py --priority 10

# Start on specific port
python testing/saturn_test_server.py --port 8081

# Start with all options
python testing/saturn_test_server.py --host 0.0.0.0 --port 8081 --priority 10
```
## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | Auto | Port to bind to (auto-selects if not specified) |
| `--priority` | `50` | SATURN service priority (lower = higher priority) |

### Test with Refactored Agents

```bash
# Start the test server
python testing/saturn_test_server.py

# Run an agent that uses SATURN discovery
python run_rcot_game.py  # Or any refactored agent
```

## Differences from openrouter_server.py

| Feature | Test Server | Full Server |
|---------|-------------|-------------|
| Models | openrouter/auto only | 343+ models |
| Model Caching | Static list | Hourly refresh |
| Lines of Code | ~300 | ~370 |
| Startup Time | < 1 second | 2-3 seconds |
| Purpose | Development/Testing | Production use |
| Features | auto-routing | multimodal, full-catalog, auto-routing |

## Why This Design?

- **Single Model:** 
- **openrouter/auto:** Provides intelligent routing (good for testing various scenarios)
- **Minimal Caching:** Reduces complexity, no stale data concerns
- **Zeroconf:** Zero-configuration discovery by agents