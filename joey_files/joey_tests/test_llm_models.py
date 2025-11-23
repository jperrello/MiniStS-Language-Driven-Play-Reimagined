"""
Comprehensive Testing Suite for LLM Model Evaluation

This test suite runs the LLMBot agent with multiple OpenRouter models through
full game runs, tracking performance metrics to enable easy comparison.

Usage:
    python testing/test_llm_models.py

Features:
- Tests multiple AI models available through OpenRouter
- Runs full game battles (not just single decisions)
- Collects comprehensive performance metrics
- Handles errors gracefully (continues testing even if individual games crash)
- Outputs human-readable results with performance comparisons

Configuration:
- Edit MODELS_TO_TEST to change which models are tested
- Edit GAMES_PER_MODEL to adjust thoroughness vs speed
- Edit ENEMY_CONFIGURATION to change difficulty
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import *
from joey_ggpa.llm_bot import LLMBot
from ggpa.prompt2 import PromptOption
from joey_setup.saturn_service_manager import SaturnServiceManager


# ============================================================================
# CONFIGS
# ============================================================================

MODELS_TO_TEST = [
    "openrouter/auto",        # auto routing to best model
    "gpt-4o-mini",            
    "google/gemini-flash-1.5" 
]

# Lower = faster, Higher = more reliable statistics
GAMES_PER_MODEL = 5

# Enemy configuration - which enemies to fight
# Options: 'j' (JawWorm), 'g' (Goblin), 'h' (HobGoblin), 'l' (Leech)
ENEMY_CONFIGURATION = "j"  # Single JawWorm (standard test)

PROMPT_STRATEGY = PromptOption.CoT  # Chain of Thought prompting

# Few-shot examples (0 = zero-shot, higher = more examples in context)
FEW_SHOT_COUNT = 0

# Show results of previous actions in prompts
SHOW_OPTION_RESULTS = False

# Use structured output format (JSON schema validation)
USE_STRUCTURED_OUTPUT = False

# Max turns before declaring game a draw (prevents infinite loops)
MAX_TURNS_PER_GAME = 50

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GameMetrics:
    game_number: int
    model_name: str
    won: bool
    turns: int
    final_player_health: int
    response_times: list[float]
    wrong_format_count: int
    wrong_range_count: int
    crashed: bool
    error_message: Optional[str] = None


@dataclass
class ModelSummary:
    model_name: str
    games_completed: int
    games_crashed: int
    win_rate: float
    avg_turns_per_game: float
    avg_response_time: float
    total_invalid_responses: int
    avg_final_health: float

    def __str__(self):
        return f"""
Model: {self.model_name}
{'=' * 60}
Games Completed: {self.games_completed}/{self.games_completed + self.games_crashed}
Win Rate: {self.win_rate:.1%}
Avg Turns per Game: {self.avg_turns_per_game:.1f}
Avg Response Time: {self.avg_response_time:.2f}s
Invalid Responses: {self.total_invalid_responses}
Avg Final Health (wins only): {self.avg_final_health:.1f}
Crash Rate: {self.games_crashed}/{self.games_completed + self.games_crashed}
{'=' * 60}
"""


def create_enemies_from_config(enemy_config: str, game_state: GameState) -> list[Enemy]:
    enemies: list[Enemy] = []
    for char in enemy_config:
        if char == 'j':
            enemies.append(JawWorm(game_state))
        elif char == 'g':
            enemies.append(Goblin(game_state))
        elif char == 'h':
            enemies.append(HobGoblin(game_state))
        elif char == 'l':
            enemies.append(Leech(game_state))
        else:
            raise ValueError(f"Unknown enemy type: {char}")
    return enemies


def run_single_game(game_number: int, model_name: str, saturn_manager: SaturnServiceManager) -> GameMetrics:
    print(f"  Game {game_number + 1}/{GAMES_PER_MODEL}: ", end="", flush=True)
    try:
        bot = LLMBot(
            model_name=model_name,
            prompt_option=PROMPT_STRATEGY,
            few_shot=FEW_SHOT_COUNT,
            show_option_results=SHOW_OPTION_RESULTS,
            use_structured_output=USE_STRUCTURED_OUTPUT,
            saturn_manager=saturn_manager
        )

        # Initialize game
        game_state = GameState(Character.IRON_CLAD, bot, ascention=0)
        enemies = create_enemies_from_config(ENEMY_CONFIGURATION, game_state)
        battle_state = BattleState(game_state,*enemies,verbose=Verbose.NO_LOG,log_filename=None)

        battle_state.initiate_log()
        turn_count = 0
        while not battle_state.ended() and turn_count < MAX_TURNS_PER_GAME:
            battle_state.take_turn()
            turn_count += 1

        # Check if game ended naturally or hit turn limit
        if turn_count >= MAX_TURNS_PER_GAME and not battle_state.ended():
            result = -1
            final_health = 0
        else:
            result = battle_state.get_end_result()
            final_health = game_state.player.health

        # Extract metrics
        won = result == 1
        response_times = bot.metadata.get("response_time", [])
        wrong_format = bot.metadata.get("wrong_format_count", 0)
        wrong_range = bot.metadata.get("wrong_range_count", 0)

        bot.cleanup()

        print(f"{'WIN' if won else 'LOSS'} - {turn_count} turns - {final_health} HP")

        return GameMetrics(
            game_number=game_number,
            model_name=model_name,
            won=won,
            turns=turn_count,
            final_player_health=final_health,
            response_times=response_times,
            wrong_format_count=wrong_format,
            wrong_range_count=wrong_range,
            crashed=False
        )

    except Exception as e:
        print(f"CRASHED - {str(e)[:50]}")
        return GameMetrics(
            game_number=game_number,
            model_name=model_name,
            won=False,
            turns=0,
            final_player_health=0,
            response_times=[],
            wrong_format_count=0,
            wrong_range_count=0,
            crashed=True,
            error_message=str(e)
        )


def test_one_model_multiple_games(model_name: str, saturn_manager: SaturnServiceManager) -> list[GameMetrics]:
    """
    Run multiple games with a specific model.

    Args:
        model_name: Model identifier
        saturn_manager: Shared SATURN service manager

    Returns:
        List of GameMetrics from all games
    """
    print(f"\nTesting Model: {model_name}")
    print("-" * 60)

    results: list[GameMetrics] = []

    for game_num in range(GAMES_PER_MODEL):
        metrics = run_single_game(game_num, model_name, saturn_manager)
        results.append(metrics)

        # Brief pause between games
        time.sleep(0.5)

    return results


def aggregate_metrics(results: list[GameMetrics]) -> ModelSummary:
    """
    Aggregate game results into summary statistics.

    Args:
        results: List of GameMetrics from all games

    Returns:
        ModelSummary with aggregated statistics
    """
    if not results:
        raise ValueError("No results to aggregate")

    model_name = results[0].model_name

    # Separate successful games from crashes
    successful_games = [r for r in results if not r.crashed]
    crashed_games = [r for r in results if r.crashed]

    games_completed = len(successful_games)
    games_crashed = len(crashed_games)

    if games_completed == 0:
        # All games crashed
        return ModelSummary(
            model_name=model_name,
            games_completed=0,
            games_crashed=games_crashed,
            win_rate=0.0,
            avg_turns_per_game=0.0,
            avg_response_time=0.0,
            total_invalid_responses=0,
            avg_final_health=0.0
        )

    # Calculate statistics from successful games
    wins = sum(1 for r in successful_games if r.won)
    win_rate = wins / games_completed

    avg_turns = sum(r.turns for r in successful_games) / games_completed

    # Aggregate all response times
    all_response_times = []
    for r in successful_games:
        all_response_times.extend(r.response_times)
    avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0.0

    total_invalid = sum(r.wrong_format_count + r.wrong_range_count for r in successful_games)

    # Average final health for wins only
    winning_games = [r for r in successful_games if r.won]
    avg_final_health = sum(r.final_player_health for r in winning_games) / len(winning_games) if winning_games else 0.0

    return ModelSummary(
        model_name=model_name,
        games_completed=games_completed,
        games_crashed=games_crashed,
        win_rate=win_rate,
        avg_turns_per_game=avg_turns,
        avg_response_time=avg_response_time,
        total_invalid_responses=total_invalid,
        avg_final_health=avg_final_health
    )


def save_results(all_results: list[GameMetrics], summaries: list[ModelSummary], output_dir: str, start_time: float):
    """
    Save detailed results and summaries to files with comprehensive formatting.

    Args:
        all_results: All game metrics
        summaries: Aggregated summaries per model
        output_dir: Directory to save results
        start_time: Test suite start time for duration calculation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate test duration
    duration = time.time() - start_time

    # Save detailed game results (JSON)
    detailed_file = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Save model summaries (JSON)
    summary_file = os.path.join(output_dir, "model_summaries.json")
    with open(summary_file, 'w') as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)

    # Generate comprehensive human-readable report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = os.path.join(output_dir, f"test_results_{timestamp}.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        _write_comprehensive_report(f, all_results, summaries, duration)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {detailed_file}")
    print(f"  - {summary_file}")
    print(f"  - {report_file}")


def _write_comprehensive_report(f, all_results: list[GameMetrics], summaries: list[ModelSummary], duration: float):
    """
    Write comprehensive, human-readable test results report.

    Args:
        f: File object to write to
        all_results: All game metrics
        summaries: Aggregated summaries per model
        duration: Total test duration in seconds
    """
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 30 + "LLM MODEL COMPARISON TEST RESULTS\n")
    f.write("=" * 100 + "\n\n")

    # Test run metadata
    f.write("TEST RUN METADATA\n")
    f.write("-" * 100 + "\n")
    f.write(f"  Test Date & Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Total Duration:          {duration:.1f}s ({duration/60:.1f} minutes)\n")
    f.write(f"  Models Tested:           {len(summaries)}\n")
    f.write(f"  Games per Model:         {GAMES_PER_MODEL}\n")
    f.write(f"  Total Games Run:         {len(all_results)}\n")
    f.write(f"  Enemy Configuration:     {ENEMY_CONFIGURATION}\n")
    f.write(f"  Prompt Strategy:         {PROMPT_STRATEGY.name}\n")
    f.write(f"  Few-Shot Examples:       {FEW_SHOT_COUNT}\n")
    f.write(f"  Structured Output:       {USE_STRUCTURED_OUTPUT}\n")
    f.write(f"  Show Option Results:     {SHOW_OPTION_RESULTS}\n")
    f.write(f"  Max Turns per Game:      {MAX_TURNS_PER_GAME}\n")
    f.write("\n")

    # ========================================================================
    # SUMMARY COMPARISON TABLE
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 38 + "SUMMARY COMPARISON\n")
    f.write("=" * 100 + "\n\n")

    # Sort summaries by win rate (descending) for ranking
    ranked_summaries = sorted(summaries, key=lambda s: (s.win_rate, -s.avg_response_time), reverse=True)

    # Table header
    f.write(f"{'Rank':<6} {'Model':<32} {'Win Rate':<12} {'Avg Time':<12} {'Invalid':<10} {'Crashes':<10} {'Avg HP':<10}\n")
    f.write("-" * 100 + "\n")

    # Table rows
    for rank, summary in enumerate(ranked_summaries, 1):
        crash_str = f"{summary.games_crashed}/{summary.games_completed + summary.games_crashed}"
        marker = " *" if rank == 1 else "  "  # Mark best performer
        f.write(f"{marker}{rank:<4} {summary.model_name:<32} {summary.win_rate:>10.1%}  "
                f"{summary.avg_response_time:>10.2f}s  {summary.total_invalid_responses:>8}  "
                f"{crash_str:>8}  {summary.avg_final_health:>8.1f}\n")

    f.write("\n")
    f.write("Legend:\n")
    f.write("  * = Best overall performer (highest win rate)\n")
    f.write("  Win Rate    = Percentage of games won (excludes crashes)\n")
    f.write("  Avg Time    = Average response time per decision\n")
    f.write("  Invalid     = Total invalid responses (wrong format + wrong range)\n")
    f.write("  Crashes     = Games that crashed / Total games attempted\n")
    f.write("  Avg HP      = Average final health points for won games only\n")
    f.write("\n")

    # ========================================================================
    # PERFORMANCE HIGHLIGHTS
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 38 + "PERFORMANCE HIGHLIGHTS\n")
    f.write("=" * 100 + "\n\n")

    if summaries:
        # Best performers in each category
        best_win_rate = max(summaries, key=lambda s: s.win_rate)
        fastest = min((s for s in summaries if s.avg_response_time > 0),
                     key=lambda s: s.avg_response_time, default=None)
        most_reliable = min(summaries, key=lambda s: s.total_invalid_responses)
        best_health = max((s for s in summaries if s.avg_final_health > 0),
                         key=lambda s: s.avg_final_health, default=None)
        most_stable = min(summaries, key=lambda s: s.games_crashed)

        f.write(f"  Highest Win Rate:       {best_win_rate.model_name:<32} ({best_win_rate.win_rate:.1%})\n")
        if fastest:
            f.write(f"  Fastest Response:       {fastest.model_name:<32} ({fastest.avg_response_time:.2f}s)\n")
        f.write(f"  Most Reliable:          {most_reliable.model_name:<32} ({most_reliable.total_invalid_responses} invalid responses)\n")
        if best_health:
            f.write(f"  Best Final Health:      {best_health.model_name:<32} ({best_health.avg_final_health:.1f} HP avg)\n")
        f.write(f"  Most Stable:            {most_stable.model_name:<32} ({most_stable.games_crashed} crashes)\n")

    f.write("\n")

    # ========================================================================
    # DETAILED MODEL BREAKDOWNS
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 35 + "DETAILED MODEL BREAKDOWNS\n")
    f.write("=" * 100 + "\n\n")

    for summary in ranked_summaries:
        _write_model_detail(f, summary, all_results)
        f.write("\n")

    # ========================================================================
    # RAW DATA SECTION
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 40 + "RAW DATA SECTION\n")
    f.write("=" * 100 + "\n\n")
    f.write("This section contains structured data for further analysis.\n\n")

    # Per-game breakdown
    f.write("PER-GAME RESULTS:\n")
    f.write("-" * 100 + "\n\n")

    for summary in ranked_summaries:
        model_results = [r for r in all_results if r.model_name == summary.model_name]
        f.write(f"Model: {summary.model_name}\n")
        f.write(f"{'Game':<6} {'Result':<8} {'Turns':<7} {'Final HP':<10} {'Avg Response':<15} {'Invalid':<10} {'Error'}\n")
        f.write("-" * 100 + "\n")

        for result in model_results:
            if result.crashed:
                result_str = "CRASH"
                error_msg = result.error_message[:50] if result.error_message else "Unknown error"
            else:
                result_str = "WIN" if result.won else "LOSS"
                error_msg = ""

            avg_resp = sum(result.response_times) / len(result.response_times) if result.response_times else 0.0
            invalid = result.wrong_format_count + result.wrong_range_count

            f.write(f"{result.game_number + 1:<6} {result_str:<8} {result.turns:<7} {result.final_player_health:<10} "
                   f"{avg_resp:<15.2f} {invalid:<10} {error_msg}\n")

        f.write("\n")

    # Summary statistics in JSON format
    f.write("SUMMARY STATISTICS (JSON):\n")
    f.write("-" * 100 + "\n")
    f.write(json.dumps([asdict(s) for s in summaries], indent=2))
    f.write("\n\n")

    # ========================================================================
    # FOOTER
    # ========================================================================
    f.write("=" * 100 + "\n")
    f.write(" " * 42 + "END OF REPORT\n")
    f.write("=" * 100 + "\n\n")

    f.write("NOTES:\n")
    f.write("  - This report is generated automatically by the LLM model testing suite\n")
    f.write("  - Results are saved with timestamps to track performance over time\n")
    f.write("  - JSON files contain machine-readable data for further analysis\n")
    f.write("  - For questions or issues, refer to testing/test_llm_models.py\n")
    f.write("\n")
    f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def _write_model_detail(f, summary: ModelSummary, all_results: list[GameMetrics]):
    """
    Write detailed breakdown for a single model.

    Args:
        f: File object to write to
        summary: Model summary statistics
        all_results: All game results (will filter for this model)
    """
    f.write(f"MODEL: {summary.model_name}\n")
    f.write("-" * 100 + "\n\n")

    # Overall statistics
    f.write("Overall Performance:\n")
    total_games = summary.games_completed + summary.games_crashed
    f.write(f"  Games Completed:        {summary.games_completed}/{total_games} ({summary.games_completed/total_games*100:.1f}%)\n")
    f.write(f"  Games Crashed:          {summary.games_crashed}/{total_games} ({summary.games_crashed/total_games*100:.1f}%)\n")
    f.write(f"  Win Rate:               {summary.win_rate:.1%} (of completed games)\n")
    f.write(f"  Average Turns/Game:     {summary.avg_turns_per_game:.1f} turns\n")
    f.write(f"  Average Response Time:  {summary.avg_response_time:.3f}s\n")
    f.write(f"  Invalid Responses:      {summary.total_invalid_responses} total\n")
    f.write(f"  Avg Final Health (wins):{summary.avg_final_health:.1f} HP\n")
    f.write("\n")

    # Get this model's results
    model_results = [r for r in all_results if r.model_name == summary.model_name]
    successful_results = [r for r in model_results if not r.crashed]

    if successful_results:
        # Response time analysis
        all_response_times = []
        for r in successful_results:
            all_response_times.extend(r.response_times)

        if all_response_times:
            min_time = min(all_response_times)
            max_time = max(all_response_times)
            f.write("Response Time Analysis:\n")
            f.write(f"  Minimum Response:       {min_time:.3f}s\n")
            f.write(f"  Maximum Response:       {max_time:.3f}s\n")
            f.write(f"  Range:                  {max_time - min_time:.3f}s\n")
            f.write("\n")

        # Game outcomes
        wins = [r for r in successful_results if r.won]
        losses = [r for r in successful_results if not r.won]

        f.write("Game Outcomes:\n")
        f.write(f"  Wins:   {len(wins)}\n")
        f.write(f"  Losses: {len(losses)}\n")

        if wins:
            avg_win_turns = sum(r.turns for r in wins) / len(wins)
            avg_win_health = sum(r.final_player_health for r in wins) / len(wins)
            f.write(f"  Avg Turns for Wins:     {avg_win_turns:.1f}\n")
            f.write(f"  Avg Final HP for Wins:  {avg_win_health:.1f}\n")

        if losses:
            avg_loss_turns = sum(r.turns for r in losses) / len(losses)
            f.write(f"  Avg Turns for Losses:   {avg_loss_turns:.1f}\n")

        f.write("\n")

        # Error analysis
        total_format_errors = sum(r.wrong_format_count for r in successful_results)
        total_range_errors = sum(r.wrong_range_count for r in successful_results)

        f.write("Error Analysis:\n")
        f.write(f"  Wrong Format Errors:    {total_format_errors}\n")
        f.write(f"  Wrong Range Errors:     {total_range_errors}\n")
        f.write(f"  Total Invalid Responses:{total_format_errors + total_range_errors}\n")
        f.write("\n")

    # Crash information
    crashed_results = [r for r in model_results if r.crashed]
    if crashed_results:
        f.write("Crash Information:\n")
        for i, result in enumerate(crashed_results, 1):
            error_msg = result.error_message[:80] if result.error_message else "Unknown error"
            f.write(f"  Crash {i} (Game {result.game_number + 1}): {error_msg}\n")
        f.write("\n")

    # Performance trend analysis (simple)
    if len(successful_results) >= 3:
        f.write("Performance Trend:\n")
        first_half = successful_results[:len(successful_results)//2]
        second_half = successful_results[len(successful_results)//2:]

        first_win_rate = sum(1 for r in first_half if r.won) / len(first_half) if first_half else 0
        second_win_rate = sum(1 for r in second_half if r.won) / len(second_half) if second_half else 0

        if second_win_rate > first_win_rate:
            trend = "IMPROVING"
        elif second_win_rate < first_win_rate:
            trend = "DECLINING"
        else:
            trend = "STABLE"

        f.write(f"  First Half Win Rate:    {first_win_rate:.1%}\n")
        f.write(f"  Second Half Win Rate:   {second_win_rate:.1%}\n")
        f.write(f"  Trend:                  {trend}\n")
        f.write("\n")

    f.write("-" * 100)


def print_comparison(summaries: list[ModelSummary]):
    """
    Print a comparison table of all models.

    Args:
        summaries: List of ModelSummary objects
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'Win Rate':<12} {'Avg Time':<12} {'Invalid':<10} {'Crashes':<10}")
    print("-" * 80)

    for summary in summaries:
        crash_str = f"{summary.games_crashed}/{summary.games_completed + summary.games_crashed}"
        print(f"{summary.model_name:<30} {summary.win_rate:>10.1%}  {summary.avg_response_time:>10.2f}s  {summary.total_invalid_responses:>8}  {crash_str:>8}")

    print("=" * 80)

    # Highlight best performers
    if summaries:
        best_win_rate = max(summaries, key=lambda s: s.win_rate)
        fastest = min(summaries, key=lambda s: s.avg_response_time if s.avg_response_time > 0 else float('inf'))
        most_reliable = min(summaries, key=lambda s: s.total_invalid_responses)

        print(f"\nBest Win Rate: {best_win_rate.model_name} ({best_win_rate.win_rate:.1%})")
        print(f"Fastest Response: {fastest.model_name} ({fastest.avg_response_time:.2f}s)")
        print(f"Most Reliable: {most_reliable.model_name} ({most_reliable.total_invalid_responses} invalid responses)")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    """Main test execution function."""
    print("=" * 80)
    print("LLM MODEL EVALUATION TEST SUITE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models to test: {len(MODELS_TO_TEST)}")
    for model in MODELS_TO_TEST:
        print(f"    - {model}")
    print(f"  Games per model: {GAMES_PER_MODEL}")
    print(f"  Total games: {len(MODELS_TO_TEST) * GAMES_PER_MODEL}")
    print(f"  Enemy config: {ENEMY_CONFIGURATION}")
    print(f"  Prompt strategy: {PROMPT_STRATEGY.name}")
    print(f"  Few-shot: {FEW_SHOT_COUNT}")
    print(f"  Structured output: {USE_STRUCTURED_OUTPUT}")
    print("=" * 80)

    # Initialize SATURN service manager once (shared across all tests)
    print("\nInitializing SATURN service manager...")
    try:
        saturn_manager = SaturnServiceManager(discovery_timeout=5.0)
    except Exception as e:
        print(f"\nERROR: Failed to initialize SATURN service manager: {e}")
        print("\nMake sure a SATURN server is running:")
        print("  python testing/saturn_test_server.py")
        return

    # Test each model
    all_results: list[GameMetrics] = []
    summaries: list[ModelSummary] = []

    start_time = time.time()

    try:
        for model_idx, model_name in enumerate(MODELS_TO_TEST, 1):
            print(f"\n[{model_idx}/{len(MODELS_TO_TEST)}] Starting tests for: {model_name}")

            model_results = test_one_model_multiple_games(model_name, saturn_manager)
            all_results.extend(model_results)

            # Aggregate and display summary
            summary = aggregate_metrics(model_results)
            summaries.append(summary)

            print(f"\nSummary for {model_name}:")
            print(f"  Win Rate: {summary.win_rate:.1%}")
            print(f"  Avg Response Time: {summary.avg_response_time:.2f}s")
            print(f"  Invalid Responses: {summary.total_invalid_responses}")

    finally:
        # Always clean up SATURN manager
        print("\nCleaning up SATURN service manager...")
        saturn_manager.close()

    elapsed_time = time.time() - start_time

    # Print final comparison
    print_comparison(summaries)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("testing", "results", f"llm_model_test_{timestamp}")
    save_results(all_results, summaries, output_dir, start_time)

    print(f"\nTotal execution time: {elapsed_time:.1f}s")
    print(f"Average time per game: {elapsed_time / (len(MODELS_TO_TEST) * GAMES_PER_MODEL):.1f}s")

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
