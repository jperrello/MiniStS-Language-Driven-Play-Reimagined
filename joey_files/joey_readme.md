# Joey's Slay the Spire LLM Agent Work - Quick Handoff

## TL;DR
This project tests if LLMs can play card games intelligently without being trained on them. Joey built infrastructure to make this actually work: **SATURN** (auto-discovers AI services), **refactored agents** (LLM bots, MCTS, RCoT), and **comprehensive testing** (compare any model against any enemy).

## The Paper's Core Idea

The "Language-Driven Play" paper asked: Can LLMs play Slay the Spire by reading card descriptions, without specialized training?

**Their findings:**
- LLMs can understand card synergies from text alone
- Backtrack agents (look-ahead search) win short-term tactics
- LLMs dominate long-term strategy (cards like "Bomb: deal 40 damage in 3 turns")
- Chain-of-Thought prompting is critical—without it, LLMs play like random agents

**Why this matters:** If LLMs can adapt to new cards automatically, you can procedurally generate game content and evaluate it without training new agents each time.

## What Joey Built

### 1. SATURN: Zero-Config AI Discovery (`joey_setup/`)

**The Problem:** Managing API keys, endpoints, and rate limits sucks. Every agent needed hardcoded credentials.

**The Solution:** SATURN uses DNS Service Discovery (same tech as AirPlay/Chromecast) to auto-find AI services on your network.

```bash
# Terminal 1: Start server (handles ALL API keys/routing)
python joey_tests/saturn_test_server.py

# Terminal 2: Run any agent (zero config needed)
python run_rcot_game.py
```

**Key files:**
- `joey_setup/saturn_server.py` - Full server with 343+ models
- `joey_setup/saturn_service_manager.py` - Client discovery system
- `joey_docs/saturn.md` - Complete documentation

**What you need to know:** Agents automatically find and connect to the highest-priority SATURN server. No API keys in your code. Ever.

### 2. Refactored Agents (`joey_ggpa/`)

Joey rebuilt the paper's agents with modern capabilities:

**LLMBot** (`llm_bot.py`):
- Uses SATURN discovery (works with 343+ OpenRouter models)
- Structured outputs = near-zero invalid responses
- Multiple prompting strategies: Chain-of-Thought, DAG reasoning
- Reasoning model support (o1, o3 models that need special handling)

**MCTSAgent** (`mcts_bot.py`):
- Monte Carlo Tree Search with UCB1 exploration
- Simulates future game states to find optimal plays
- From the paper: better than backtrack for complex scenarios

**RCotAgent** (`rcot_agent.py`):
- "Reasoning Chain-of-Thought" - experimental agent
- Combines LLM reasoning with structured decision-making
- Configurable via `RCotConfig`

**Critical insight from the paper:** The LLM without Chain-of-Thought performed identically to random play. With CoT, it outperformed search-based agents on strategic decisions. Joey's implementation makes CoT the default.

### 3. Testing Infrastructure (`joey_tests/`)

**`test_all_agents.py`** - The main event:
```bash
python joey_tests/test_all_agents.py --agents llm mcts backtrack --enemies jj --games 10
```
- Run any agent vs any enemy configuration
- Collect win rates, turn counts, HP remaining, response times
- Compare multiple agents head-to-head

**`test_llm_models.py`** - Model comparison suite:
- Test GPT-4, Claude, Gemini, Llama—any of 343+ models
- Configurable at the top of the file (enemies, games per model, prompt strategy)
- Generates detailed JSON results + human-readable reports
- Auto-saves to `testing/results/` with timestamps

**Enemy configs:**
- `j` = one JawWorm
- `jj` = two JawWorms
- `g` = Goblin
- Mix/match: `jg`, `jgg`, etc.

## How The Code Extends The Paper

**Paper limitations Joey fixed:**

1. **Model flexibility:** Paper used GPT-3.5 only. Joey's SATURN lets you test 343+ models in minutes.

2. **Prompt engineering:** Paper tested 4 prompt styles. Joey's agents support CoT, DAG, few-shot examples, and structured outputs (reducing invalid responses from 1.38% to ~0%).

3. **Long-term reasoning:** Paper showed LLMs excel at cards needing multi-turn planning. Joey's RCotAgent and structured prompting push this further.

4. **Evaluation at scale:** Paper ran 50 games per scenario manually. Joey's test suite runs hundreds of games across multiple models/agents/enemies automatically.

## Start Here

**Day 1 - Get it running:**
1. Add OpenRouter API key to `.env` (see `joey_docs/saturn_start.md`)
2. Run: `python testing/saturn_test_server.py`
3. Run: `python joey_tests/test_all_agents.py --agents llm random --enemies j --games 5`

**Day 2 - Understand the flow:**
1. Read `joey_docs/saturn.md` (how services auto-discover)
2. Read `joey_docs/model_testing.md` (how to compare models)
3. Look at `joey_ggpa/llm_bot.py` lines 1-100 (prompting logic)

**Day 3 - Run experiments:**
1. Edit `joey_tests/test_llm_models.py` config (top of file)
2. Test different models against different enemies
3. Check results in `testing/results/`

## Documentation Structure

- `joey_docs/*.md` - How to use everything
- `joey_ggpa/*.py` - Agent implementations
- `joey_setup/*.py` - SATURN infrastructure
- `joey_tests/*.py` - Testing + examples

