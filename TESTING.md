# Testing Guide

Reqs:
```bash
pip install pandas tqdm seaborn matplotlib joblib openai dotenv
```

## TLDR;

**Test Generated Cards (GIGL):**
```bash
python evaluation/evaluate_card_gen.py 10 4 20 h cot-claude --gigl-dir GIGL/generated_cards --name gigl_test
```
Note: this uses money to pay for claude.
It runs 10 sims for each 20 cards with 4 threads
If you want to test for free:
```bash
python evaluation/evaluate_card_gen.py 10 4 20 h bt5 --gigl-dir GIGL/generated_cards --name gigl_test
```

**Default Agent Comparison Test:**
```bash
python evaluation/evaluate_bot.py 50 4 0 h r bt3 none cot rcot --name agent_comparison --time
```

**Paid Agent Test (this is $$$, but also the most important command):**
```bash
python evaluation/evaluate_bot.py 25 4 0 h rcot-gpt41 rcot-openrouter-auto rcot-claude rcot-gemini mcts rndm --name premium --time
```

**Generate Statistics Table:**
```bash
python evaluation/generate_table.py evaluation_results/<your_test_directory>/results.csv evaluation_results/<your_test_directory>/execution_times.json
```

**Plot Results:**
```bash
# For bot evaluation results (group by BotName):
python evaluation/plot_evaluation.py evaluation_results/<your_test_directory>/results.csv BotName

# For card generation results (group by CardName):
python evaluation/plot_evaluation.py evaluation_results/card_gen_<name>_enemies_<enemies>_<test_count>_<bot>/results.csv CardName
```
## How to run

### Available Bots (All use group 3's agents and not the paper's)
- Baseline: `rndm`, `basic`, `r` (random)
- None (Minimal prompting - PromptOption.NONE - asks LLM to pick action index without explanation):
  - `none` (default: GPT-4.1)
  - Premium: `none-gpt41`, `none-openrouter-auto`, `none-claude`, `none-gemini`
  - Free: `none-llama-free`, `none-qwen-free`, `none-nemotron-free`, `none-gpt-oss-free`, `none-deepseek-free`
- Backtrack: `bt<depth>`, `bts<depth>` (e.g., `bt3`, `bts5`)
- MCTS: `mcts`, `mcts-<iterations>` (e.g., `mcts-200`)
- CoT (Chain-of-Thought):
  - `cot` (default: GPT-4.1)
  - Premium: `cot-gpt41`, `cot-openrouter-auto`, `cot-claude`, `cot-gemini`
  - Free: `cot-llama-free`, `cot-qwen-free`, `cot-nemotron-free`, `cot-gpt-oss-free`, `cot-deepseek-free`
- RCoT (Reverse Chain-of-Thought):
  - `rcot` (default: OpenRouter auto-routing)
  - Premium: `rcot-gpt41`, `rcot-openrouter-auto`, `rcot-claude`, `rcot-gemini`
  - Free: `rcot-llama-free`, `rcot-qwen-free`, `rcot-nemotron-free`, `rcot-gpt-oss-free`, `rcot-deepseek-free`
- Legacy GPT: `legacy-gpt-<model>-<prompt>` (e.g., `legacy-gpt-t3.5-cot`)

### Available Enemies (Global)
Enemy configuration string (e.g., `gsl` for Goblin, SimpleEnemy, Leech):
- `h` = HobGoblin (from paper: 22 damage attack, 10 damage + 10 block, probably what you want to run)
- `g` = Goblin
- `l` = Leech
- `j` = JawWorm

---
### 1. Card Generation Evaluation (`evaluate_card_gen.py`)

Evaluates the impact of generated or custom cards on bot performance.

**Basic Usage:**
```bash
python evaluation/evaluate_card_gen.py <test_count> <thread_count> <gen_count> <enemies> <bot>
```

**Arguments:**
- `test_count`: Number of test simulations per card
- `thread_count`: Number of parallel threads
- `gen_count`: Number of cards to generate/test
- `enemies`: Enemy configuration string (same as evaluate_bot.py)
- `bot`: Bot name to use for evaluation

**Optional Flags:**
- `--name <name>`: Custom name for the test run
- `--dir <directory>`: Custom directory for results
- `--log`: Enable logging
- `--gigl-dir <directory>`: Directory containing GIGL JSON card files (instead of random generation)

**Example (Random Cards):**
```bash
python evaluation/evaluate_card_gen.py 50 4 10 ghl mcts --name random_cards
```

**Example (GIGL Cards):**
```bash
python evaluation/evaluate_card_gen.py 50 4 10 ghl mcts --gigl-dir path/to/cards --name gigl_test
```

**Output:**
Results are saved to `evaluation_results/card_gen_<name>_enemies_<enemies>_<test_count>_<bot>/`
- `results.csv`: Columns: BotName, CardName, PlayerHealth, Win

---
### 2. Bot Evaluation (`evaluate_bot.py`)

Evaluates different bot agents against specific scenarios and enemies.

**Basic Usage:**
```bash
python evaluation/evaluate_bot.py <test_count> <thread_count> <scenario> <enemies> <bot1> [bot2 ...]
```

**Arguments:**
- `test_count`: Number of test simulations to run per bot
- `thread_count`: Number of parallel threads to use (this makes it faster but leads to pickling if my code breaks)
- `scenario`: Scenario index (0-4) These scenarios are the same as the paper: starter, batter-stimulate, bomb, tolerate, and harm; numbered accordingly.
- `enemies`: Enemy configuration string (see Available Enemies above)
- `bot1`, `bot2`, etc.: Bot names to evaluate (see Available Bots above)

**Optional Flags:**
- `--name <name>`: Custom name for the test run
- `--dir <directory>`: Custom directory for results
- `--log`: Enable logging
- `--anonymize`: Anonymize card names in the scenario
- `--time`: Track execution time for each bot

**Example:**
```bash
python evaluation/evaluate_bot.py 100 4 0 s none basic mcts --name my_test --log
```

**Output:**
Results are saved to `evaluation_results/<name>_<scenario>_enemies_<enemies>_<test_count>_boteval/`
- `results.csv`: Simulation results with columns: BotName, PlayerHealth, Win
- `execution_times.json`: Execution times per bot (if `--time` flag used)
- Individual log files for each simulation (if `--log` flag used)

---




### 3. Plot Evaluation Results (`plot_evaluation.py`)

Generates statistical plots from evaluation results.

**Basic Usage:**
```bash
python evaluation/plot_evaluation.py <filename> <by_column>
```

**Arguments:**
- `filename`: Path to the CSV results file
- `by_column`: Column to group by (e.g., `BotName`, `CardName`)

**Optional Flags:**
- `--maxx <value>`: Maximum x-axis value for the plot

**Example:**
```bash
python evaluation/plot_evaluation.py evaluation_results/my_test/results.csv BotName --maxx 80
```

**Output:**
- Displays histogram with KDE showing distribution of PlayerHealth by the specified column
- Prints mean PlayerHealth for each group

---
## Output Directory Structure

All results are saved in the `evaluation_results/` directory with the following structure:
```
evaluation_results/
├── <test_name>_<scenario>_enemies_<enemies>_<test_count>_boteval/
│   ├── results.csv
│   ├── execution_times.json (if --time used)
│   ├── <id>_<bot_name>.log (if --log used)
│   └── <id>_<bot_name>_history (ChatGPT bots only)
└── card_gen_<name>_enemies_<enemies>_<test_count>_<bot>/
    └── results.csv
```

## Common Workflows

### Workflow 1: Test GIGL Generated Cards
```bash
python evaluation/evaluate_card_gen.py 10 4 20 s cot-claude --gigl-dir GIGL/generated_cards --name gigl_test
python evaluation/plot_evaluation.py evaluation_results/card_gen_gigl_test_*/results.csv CardName
```

### Workflow 2: Compare Multiple Bots
```bash
python evaluation/evaluate_bot.py 100 8 0 ghl none basic bt3 mcts --name bot_comparison --time
python evaluation/plot_evaluation.py evaluation_results/bot_comparison_*/results.csv BotName
```


### Workflow 3: Analyze Game Logs
```bash
python evaluation/evaluate_bot.py 100 4 0 ghl bt5 --name log_analysis --log
python evaluation/plot_property.py evaluation_results/log_analysis_*
```

