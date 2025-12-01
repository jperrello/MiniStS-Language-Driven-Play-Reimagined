"""
Example Test Configurations for LLM Model Testing

This file contains pre-configured test scenarios that you can copy into
test_llm_models.py to quickly run different types of evaluations.

Usage:
1. Copy the configuration you want
2. Paste it into the CONFIGURATION SECTION of test_llm_models.py
3. Run the test suite
"""

# ============================================================================
# SCENARIO 1: Quick Comparison (Fast)
# ============================================================================
# Use case: Quick sanity check of top models
# Time: ~3-5 minutes
# Best for: Daily testing, CI/CD pipelines

QUICK_COMPARISON = {
    "MODELS_TO_TEST": [
        "openrouter/auto",
        "gpt-4o-mini"
    ],
    "GAMES_PER_MODEL": 3,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 2: Comprehensive Evaluation (Thorough)
# ============================================================================
# Use case: Production model selection
# Time: ~20-30 minutes
# Best for: Major decisions, final testing before deployment

COMPREHENSIVE_EVAL = {
    "MODELS_TO_TEST": [
        "openrouter/auto",
        "gpt-4o-mini",
        "gpt-4o",
        "google/gemini-flash-1.5",
        "google/gemini-pro-1.5",
        "anthropic/claude-3.5-sonnet"
    ],
    "GAMES_PER_MODEL": 10,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 3: Reasoning Model Evaluation
# ============================================================================
# Use case: Testing models with enhanced reasoning capabilities
# Time: ~15-20 minutes (reasoning models are slower)
# Best for: Complex game scenarios, strategic planning

REASONING_MODELS = {
    "MODELS_TO_TEST": [
        "o1-mini",
        "o1-preview",
        "gpt-4o"  # For comparison
    ],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "jj",  # Harder scenario
    "PROMPT_STRATEGY": "PromptOption.CoT",  # Reasoning models work well with CoT
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 5: Provider Comparison
# ============================================================================
# Use case: Compare performance across different AI providers
# Time: ~10-15 minutes
# Best for: Understanding provider strengths/weaknesses

PROVIDER_COMPARISON = {
    "MODELS_TO_TEST": [
        "gpt-4o-mini",                    # OpenAI
        "anthropic/claude-3.5-sonnet",    # Anthropic
        "google/gemini-flash-1.5",        # Google
        "meta-llama/llama-3.1-70b-instruct"  # Meta
    ],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 6: Prompt Strategy Comparison
# ============================================================================
# Use case: Find best prompt strategy for a specific model
# Instructions: Run test suite multiple times, changing PROMPT_STRATEGY each time
# Time: ~5 minutes per strategy
# Best for: Optimizing prompt engineering

# Run 1: No prompting
PROMPT_NONE = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.NONE",
    "FEW_SHOT_COUNT": 0,
}

# Run 2: Chain of Thought
PROMPT_COT = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# Run 3: DAG reasoning
PROMPT_DAG = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.DAG",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 7: Few-Shot Learning Evaluation
# ============================================================================
# Use case: Test impact of few-shot examples
# Instructions: Run test suite multiple times with different FEW_SHOT_COUNT
# Time: ~5 minutes per configuration
# Best for: Understanding context window utilization

# Run 1: Zero-shot
FEW_SHOT_ZERO = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# Run 2: Few-shot (3 examples)
FEW_SHOT_THREE = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 3,
}

# Run 3: Many-shot (5 examples)
FEW_SHOT_FIVE = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 5,
}

# ============================================================================
# SCENARIO 8: Structured Output Testing
# ============================================================================
# Use case: Evaluate impact of structured outputs on reliability
# Time: ~5 minutes per run
# Best for: Reducing invalid response rates

# Run 1: Without structured output
NO_STRUCTURED_OUTPUT = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
    "USE_STRUCTURED_OUTPUT": False,
}

# Run 2: With structured output
WITH_STRUCTURED_OUTPUT = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
    "USE_STRUCTURED_OUTPUT": True,
}

# ============================================================================
# SCENARIO 9: Difficulty Scaling
# ============================================================================
# Use case: Test model performance on increasingly difficult scenarios
# Instructions: Run test suite multiple times with different enemy configs
# Time: ~5 minutes per difficulty
# Best for: Understanding model capabilities under pressure

# Easy: Single weak enemy
DIFFICULTY_EASY = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "j",  # Single JawWorm
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# Medium: Multiple enemies
DIFFICULTY_MEDIUM = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "jj",  # Two JawWorms
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# Hard: Many enemies
DIFFICULTY_HARD = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 5,
    "ENEMY_CONFIGURATION": "jjj",  # Three JawWorms
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 10: Regression Testing
# ============================================================================
# Use case: Verify current production model still performs well
# Time: ~5-10 minutes
# Best for: Pre-deployment validation, monitoring

REGRESSION_TEST = {
    "MODELS_TO_TEST": [
        "gpt-4o-mini"  # Replace with your production model
    ],
    "GAMES_PER_MODEL": 10,  # Higher sample size for reliability
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 11: A/B Testing
# ============================================================================
# Use case: Compare two specific models in detail
# Time: ~10 minutes
# Best for: Final decision between two candidates

AB_TEST = {
    "MODELS_TO_TEST": [
        "gpt-4o-mini",      # Option A
        "google/gemini-flash-1.5"  # Option B
    ],
    "GAMES_PER_MODEL": 10,  # Larger sample for statistical significance
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# SCENARIO 12: Stress Testing
# ============================================================================
# Use case: Test model stability over many games
# Time: ~30-60 minutes
# Best for: Identifying edge cases, memory leaks, consistency

STRESS_TEST = {
    "MODELS_TO_TEST": ["gpt-4o-mini"],
    "GAMES_PER_MODEL": 50,  # Many games
    "ENEMY_CONFIGURATION": "j",
    "PROMPT_STRATEGY": "PromptOption.CoT",
    "FEW_SHOT_COUNT": 0,
}

# ============================================================================
# How to Use These Configurations
# ============================================================================

"""
EXAMPLE USAGE:

1. Choose a scenario from above (e.g., QUICK_COMPARISON)

2. Open test_llm_models.py

3. Replace the configuration section with your chosen scenario:

   # Before:
   MODELS_TO_TEST = ["openrouter/auto", ...]
   GAMES_PER_MODEL = 5

   # After (for quick comparison):
   MODELS_TO_TEST = ["openrouter/auto", "gpt-4o-mini"]
   GAMES_PER_MODEL = 3
   ENEMY_CONFIGURATION = "j"
   PROMPT_STRATEGY = PromptOption.CoT
   FEW_SHOT_COUNT = 0

4. Run the test suite:
   python testing/test_llm_models.py

5. Compare results in testing/results/llm_model_test_[timestamp]/

TIPS:
- Start with QUICK_COMPARISON to verify everything works
- Use COMPREHENSIVE_EVAL for important decisions
- Run REGRESSION_TEST before major deployments
- Use A/B testing for final model selection
- Archive results from each configuration for comparison
"""

# ============================================================================
# Custom Configuration Template
# ============================================================================

CUSTOM_CONFIG_TEMPLATE = {
    "MODELS_TO_TEST": [
        # Add your models here
        # Examples:
        # "openrouter/auto",
        # "gpt-4o-mini",
        # "google/gemini-flash-1.5",
        # "anthropic/claude-3.5-sonnet",
        # "meta-llama/llama-3.1-70b-instruct",
    ],
    "GAMES_PER_MODEL": 5,  # Adjust based on time available
    "ENEMY_CONFIGURATION": "j",  # "j", "jj", "g", "gg", etc.
    "PROMPT_STRATEGY": "PromptOption.CoT",  # NONE, CoT, CoT_rev, DAG
    "FEW_SHOT_COUNT": 0,  # 0-5 typically
    "SHOW_OPTION_RESULTS": False,
    "USE_STRUCTURED_OUTPUT": False,
    "MAX_TURNS_PER_GAME": 50,
}
