# Joey's Slay the Spire LLM Agent Work

## TL;DR
Test custom Slay the Spire cards against AI agents (LLM, MCTS, RCoT, Backtrack, Random). Uses **SATURN** for zero-config AI discovery. Simplified structure for easy testing.

## Quick Start

### 1. Setup SATURN Server (One-Time)
```bash
# Add OpenRouter API key to .env file
echo "OPENROUTER_API_KEY=your-key-here" >> .env

# Start SATURN server
python joey_files/joey_setup/saturn_server.py
```

### 2. Test Your Cards
```bash
# Test with defaults (LLM agent, starter deck, 5 games)
python joey_files/joey_tests/test_cards.py

# Test custom cards against multiple agents
python joey_files/joey_tests/test_cards.py --cards my_cards.json --agent mcts,llm,random --games 10

# Compare all agents
python joey_files/joey_tests/test_cards.py --agent all --scenario basic --enemy jj --games 5
```

## File Structure

```
joey_files/
├── joey_ggpa/          # AI agent implementations
│   ├── mcts_bot.py     # Monte Carlo Tree Search
│   ├── llm_bot.py      # LLM-based agent
│   └── rcot_agent.py   # Reflective Chain-of-Thought
├── joey_setup/         # SATURN service discovery
│   ├── saturn_server.py          # Server (handles API keys)
│   └── saturn_service_manager.py # Client discovery
├── joey_tests/         # Testing infrastructure
│   └── test_cards.py   # Unified testing suite
└── joey_readme.md      # This file
```

## The Core Agents

### MCTSAgent (`joey_ggpa/mcts_bot.py`)
Monte Carlo Tree Search using UCB1 exploration. Simulates future game states through random rollouts.

**From the paper:** Better than backtrack for complex scenarios by exploring broader state spaces.

**Configuration:**
```python
MCTSAgent(iterations=100, exploration=0.5)
```

### LLMBot (`joey_ggpa/llm_bot.py`)
Uses SATURN discovery to access 343+ AI models through OpenRouter.

**Key Features:**
- Zero-configuration service discovery
- Structured outputs (near-zero invalid responses)
- Multiple prompting strategies (CoT, DAG, etc.)
- Reasoning model support (o1, o3)

**Supported Models:**
- `openrouter/auto` - Intelligent routing (recommended)
- GPT-4 family: `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- Reasoning models: `o1-preview`, `o1-mini`, `o3-mini`
- Claude: `claude-3.5-sonnet`, `claude-3-opus`
- Gemini: `gemini-pro-1.5`, `gemini-flash-1.5`
- Llama: `llama-3.1-70b-instruct`

**Critical insight from the paper:** LLMs without Chain-of-Thought perform identically to random play. With CoT, they outperform search-based agents on strategic decisions.

### RCotAgent (`joey_ggpa/rcot_agent.py`)
Reflective Chain-of-Thought from the "Language-Driven Play" paper (Bateni & Whitehead, FDG 2024).

**Features:**
- Three prompt options: none, cot, rcot
- Card name anonymization for generalization
- Configurable via `RCotConfig`

## SATURN: Zero-Config AI Discovery

**SATURN eliminates API key management.** Agents automatically discover AI services on your local network via DNS-SD (same tech as AirPlay/Chromecast).

### How It Works

**Service Discovery:**
- Service Type: `_saturn._tcp.local.`
- Discovery Method: DNS-SD subprocess commands
- Registration: Servers use `dns-sd -R` to announce themselves
- Priority System: Lower numbers = higher priority (default: 50)

**Architecture:**
- Agents use `SaturnServiceManager` to discover services
- Continuous background monitoring for services
- Automatic connection to highest-priority service
- Automatic failover when services go offline
- No API keys in agent code

### Setup

1. **Add your API key to `.env`** (one-time):
   ```bash
   echo "OPENROUTER_API_KEY=your-key-here" >> .env
   ```

2. **Start the SATURN server**:
   ```bash
   python joey_files/joey_setup/saturn_server.py
   ```

   The server auto-announces via DNS-SD and caches 343+ models from OpenRouter.

3. **Run any agent** - they automatically discover and connect:
   ```bash
   python joey_files/joey_tests/test_cards.py --agent llm
   ```

### Advanced Configuration

```bash
# Start with high priority (preferred by clients)
python joey_files/joey_setup/saturn_server.py --priority 10

# Start on specific port
python joey_files/joey_setup/saturn_server.py --port 8081
```

## Testing Infrastructure

### test_cards.py - Unified Testing Suite

Replace all previous test files with a single, parameter-driven interface.

**Parameters:**
- `--agent`: Which agent(s) to test (llm, mcts, rcot, backtrack, random, all)
- `--model`: Which LLM model to use (for LLM/RCoT agents)
- `--cards`: Path to custom cards JSON file
- `--scenario`: Predefined scenarios (starter, basic, scaling, vigor, lowhp)
- `--enemy`: Enemy configuration (j, jj, g, gg, jg, etc.)
- `--games`: Number of games to run (default: 5)
- `--verbose`: Show detailed game logs
- `--output`: Path to save results JSON

**Examples:**
```bash
# Test custom cards against LLM agent using GPT-4o
python joey_files/joey_tests/test_cards.py --agent llm --model gpt-4o --cards my_cards.json --enemy jj --games 10

# Test multiple agents against starter deck
python joey_files/joey_tests/test_cards.py --agent mcts,llm,random --scenario starter --enemy j --games 5

# Quick comparison of all agents
python joey_files/joey_tests/test_cards.py --agent all --scenario basic --games 3

# Test with verbose output
python joey_files/joey_tests/test_cards.py --agent llm --scenario vigor --enemy j --games 1 --verbose
```

**Enemy Configurations:**
- `j` - Single JawWorm (standard difficulty)
- `jj` - Two JawWorms
- `g` - Single Goblin
- `gg` - Two Goblins
- Mix: `jg`, `jgg`, etc.

**Scenarios:**
- `starter` - Default Slay the Spire starting deck (20 HP)
- `basic` - Mixed offensive/defensive cards (18 HP)
- `scaling` - Upgrade-focused strategy (16 HP)
- `vigor` - Temporary damage boost focus (15 HP)
- `lowhp` - Low health, defensive play (8 HP)

**Custom Cards:**
Create a JSON file with your card deck:
```json
{
  "cards": [
    "Strike",
    "Strike",
    "Defend",
    "Defend",
    "Bash",
    "YourCustomCard"
  ],
  "starting_hp": 20
}
```

### Output

The test suite provides clear, ranked results:

```
================================================================================
TEST RESULTS SUMMARY
================================================================================

Agent                Games   Wins  Win %   Avg Turns  Avg Time   Avg Dmg %
--------------------------------------------------------------------------------
LLM-4o-mini           5       4    80.00%      7.20      1.15s     95.50%
MCTS-100              5       3    60.00%      8.50      2.30s     85.20%
Random                5       1    20.00%      6.00      0.05s     45.30%

Best performing agent: LLM-4o-mini with 80.00% win rate
```

## The Paper's Core Idea

The "Language-Driven Play" paper asked: **Can LLMs play Slay the Spire by reading card descriptions, without specialized training?**

**Their findings:**
- LLMs can understand card synergies from text alone
- Backtrack agents win short-term tactics
- LLMs dominate long-term strategy (e.g., "Bomb: deal 40 damage in 3 turns")
- **Chain-of-Thought prompting is critical** - without it, LLMs play like random agents

**Why this matters:** If LLMs can adapt to new cards automatically, you can procedurally generate game content and evaluate it without training new agents each time.

## How This Code Extends The Paper

**Paper limitations we fixed:**

1. **Model flexibility:** Paper used GPT-3.5 only. SATURN lets you test 343+ models instantly.

2. **Prompt engineering:** Paper tested 4 prompt styles. Our agents support CoT, DAG, few-shot examples, and structured outputs (reducing invalid responses from 1.38% to ~0%).

3. **Long-term reasoning:** Paper showed LLMs excel at multi-turn planning. RCotAgent and structured prompting push this further.

4. **Evaluation at scale:** Paper ran 50 games per scenario manually. Our test suite runs hundreds of games across multiple models/agents/enemies automatically.

## Troubleshooting

### SATURN Server Won't Start
- Check `.env` file exists with `OPENROUTER_API_KEY`
- Ensure dns-sd is available (install Bonjour on Windows)
- Try specifying a port: `--port 8081`

### Agents Can't Find SATURN Server
- Check server is running: `python joey_files/joey_setup/saturn_server.py`
- Verify DNS-SD is working: `dns-sd -B _saturn._tcp local`
- Increase discovery timeout in agent code

### Invalid LLM Responses
- Enable structured outputs: `--use-structured-output`
- Try a different model: `--model gpt-4o`
- Check prompt strategy: default is CoT (recommended)

## Next Steps

1. **Create custom cards** - Build a JSON file with your card deck
2. **Test against all agents** - See which strategies work best
3. **Experiment with models** - Try different LLMs via `--model`
4. **Analyze results** - Use `--output` to save detailed metrics
5. **Read the paper** - "Language-Driven Play" by Bateni & Whitehead (FDG 2024)

## Credits

**Original Research:** "Language-Driven Play: Large Language Models as Game-Playing Agents in Slay the Spire" by Bateni & Whitehead (FDG 2024)

**Infrastructure:** SATURN service discovery, refactored agents, unified testing suite

**Game:** Slay the Spire by MegaCrit
