# LLM Model Testing Suite

## Overview

The LLM Model Testing Suite (`test_llm_models.py`) is a comprehensive framework for evaluating different AI models available through OpenRouter. It runs full game battles with the LLMBot agent and collects detailed performance metrics to enable informed model selection.

## Quick Start

### Prerequisites

1. **SATURN Server Running**: The test suite requires a SATURN server for AI model access.
   ```bash
   python testing/saturn_test_server.py
   ```

2. **Environment Setup**: Ensure you have a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

### Running Tests

Basic usage:
```bash
python testing/test_llm_models.py
```

This will:
1. Initialize SATURN service discovery
2. Test each configured model with 5 games each
3. Display progress and results in real-time
4. Save detailed results to `testing/results/llm_model_test_[timestamp]/`

## Configuration

All configuration is at the top of `test_llm_models.py`. Edit these constants to customize testing:

### Model Selection

```python
MODELS_TO_TEST = [
    "openrouter/auto",        # Intelligent routing
    "gpt-4o-mini",            # Cost-effective GPT
    "google/gemini-flash-1.5" # Fast Gemini
]
```

**Available Models** (343 via OpenRouter):
- OpenRouter routing: `openrouter/auto`
- GPT-4 family: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- Reasoning models: `o1-preview`, `o1-mini`, `o3-mini`
- Claude: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`
- Gemini: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`
- Llama: `meta-llama/llama-3.1-70b-instruct`

### Test Parameters

```python
GAMES_PER_MODEL = 5           # Games to run per model
ENEMY_CONFIGURATION = "j"      # Which enemies to fight
PROMPT_STRATEGY = PromptOption.CoT  # Prompting strategy
FEW_SHOT_COUNT = 0            # Number of few-shot examples
SHOW_OPTION_RESULTS = False   # Show action outcomes
USE_STRUCTURED_OUTPUT = False # Use JSON schema validation
MAX_TURNS_PER_GAME = 50       # Turn limit per game
```

### Enemy Configurations

- `"j"` - Single JawWorm (standard difficulty)
- `"jj"` - Two JawWorms (harder)
- `"g"` - Single Goblin
- `"gg"` - Two Goblins
- Mix and match: `"jg"`, `"jgg"`, etc.

## Metrics Collected

### Per-Game Metrics
- **Won**: (boolean)
- **Turns Taken**
- **HP Remaining**: HP remaining (for wins)
- **Response Times**: T
- **Wrong Format Count**: Parsing failures
- **Wrong Range Count**: Out-of-range selections
- **Crashed**:  (boolean)
- **Error Message**

### Aggregated Model Metrics
- **Win Rate**
- **Avg Turns per Game** 
- **Avg Response Time**
- **Total Invalid Responses**: Sum of format/range errors
- **Avg Final Health**: 
- **Crash Rate**: 

## Output Files

Results are saved to `testing/results/llm_model_test_[timestamp]/`:

### 1. `detailed_results.json`
Complete per-game metrics for all models:
```json
[
  {
    "game_number": 0,
    "model_name": "gpt-4o-mini",
    "won": true,
    "turns": 8,
    "final_player_health": 64,
    "response_times": [1.23, 0.98, ...],
    "wrong_format_count": 0,
    "wrong_range_count": 0,
    "crashed": false
  },
  ...
]
```

### 2. `model_summaries.json`
Aggregated statistics per model:
```json
[
  {
    "model_name": "gpt-4o-mini",
    "games_completed": 5,
    "games_crashed": 0,
    "win_rate": 0.8,
    "avg_turns_per_game": 7.2,
    "avg_response_time": 1.15,
    "total_invalid_responses": 0,
    "avg_final_health": 58.5
  },
  ...
]
```

### 3. `test_results_[timestamp].txt`
Comprehensive human-readable report with detailed analysis:
```
====================================================================================================
                              LLM MODEL COMPARISON TEST RESULTS
====================================================================================================

TEST RUN METADATA
----------------------------------------------------------------------------------------------------
  Test Date & Time:        2025-11-21 15:30:45
  Total Duration:          145.3s (2.4 minutes)
  Models Tested:           3
  Games per Model:         5
  Enemy Configuration:     j
  Prompt Strategy:         CoT

====================================================================================================
                                      SUMMARY COMPARISON
====================================================================================================

Rank   Model                            Win Rate     Avg Time     Invalid    Crashes    Avg HP
----------------------------------------------------------------------------------------------------
 *1    gpt-4o-mini                           80.0%        1.15s         0       0/5      58.5
  2    openrouter/auto                       60.0%        1.23s         2       0/5      52.0
...

PERFORMANCE HIGHLIGHTS
----------------------------------------------------------------------------------------------------
  Highest Win Rate:       gpt-4o-mini                      (80.0%)
  Fastest Response:       openrouter/auto                  (1.15s)
  Most Reliable:          gpt-4o-mini                      (0 invalid responses)

DETAILED MODEL BREAKDOWNS
----------------------------------------------------------------------------------------------------
[Complete breakdown per model with response time analysis, game outcomes, error analysis,
 crash information, and performance trends]

RAW DATA SECTION
----------------------------------------------------------------------------------------------------
[Per-game results table and JSON summaries for further analysis]
```

For complete format documentation, see: `testing/RESULTS_FORMAT.md`
