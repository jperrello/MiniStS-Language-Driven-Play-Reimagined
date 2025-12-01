"""
Demo script to show the results file format without running full tests.
This generates a sample results file with mock data to demonstrate the output format.
"""

import sys
import os
from datetime import datetime
from dataclasses import dataclass
import random

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the data structures from test_llm_models
from testing.test_llm_models import GameMetrics, ModelSummary, _write_comprehensive_report


def generate_mock_results():
    """Generate mock test results for demonstration."""
    models = [
        "openrouter/auto",
        "gpt-4o-mini",
        "google/gemini-flash-1.5"
    ]

    all_results = []

    for model_idx, model in enumerate(models):
        # Generate 5 mock games per model
        for game_num in range(5):
            # Vary performance by model to show differences
            if model == "openrouter/auto":
                won = game_num < 4  # 80% win rate
                turns = random.randint(8, 15)
                final_health = random.randint(45, 70) if won else 0
                response_times = [random.uniform(0.8, 1.5) for _ in range(turns)]
                wrong_format = random.randint(0, 1)
                wrong_range = random.randint(0, 1)
            elif model == "gpt-4o-mini":
                won = game_num < 3  # 60% win rate
                turns = random.randint(10, 18)
                final_health = random.randint(35, 60) if won else 0
                response_times = [random.uniform(0.5, 1.0) for _ in range(turns)]
                wrong_format = random.randint(0, 2)
                wrong_range = random.randint(0, 1)
            else:  # gemini-flash
                won = game_num < 3  # 60% win rate
                turns = random.randint(9, 16)
                final_health = random.randint(30, 55) if won else 0
                response_times = [random.uniform(0.4, 0.9) for _ in range(turns)]
                wrong_format = random.randint(0, 3)
                wrong_range = random.randint(1, 2)

            all_results.append(GameMetrics(
                game_number=game_num,
                model_name=model,
                won=won,
                turns=turns,
                final_player_health=final_health,
                response_times=response_times,
                wrong_format_count=wrong_format,
                wrong_range_count=wrong_range,
                crashed=False
            ))

    # Generate summaries
    summaries = []
    for model in models:
        model_results = [r for r in all_results if r.model_name == model]

        wins = sum(1 for r in model_results if r.won)
        win_rate = wins / len(model_results)
        avg_turns = sum(r.turns for r in model_results) / len(model_results)

        all_response_times = []
        for r in model_results:
            all_response_times.extend(r.response_times)
        avg_response_time = sum(all_response_times) / len(all_response_times)

        total_invalid = sum(r.wrong_format_count + r.wrong_range_count for r in model_results)

        winning_games = [r for r in model_results if r.won]
        avg_final_health = sum(r.final_player_health for r in winning_games) / len(winning_games) if winning_games else 0.0

        summaries.append(ModelSummary(
            model_name=model,
            games_completed=len(model_results),
            games_crashed=0,
            win_rate=win_rate,
            avg_turns_per_game=avg_turns,
            avg_response_time=avg_response_time,
            total_invalid_responses=total_invalid,
            avg_final_health=avg_final_health
        ))

    return all_results, summaries


def main():
    """Generate sample results file."""
    print("Generating sample results file...")

    # Create results directory if needed
    results_dir = os.path.join("testing", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate mock data
    all_results, summaries = generate_mock_results()

    # Create sample results file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(results_dir, f"SAMPLE_test_results_{timestamp}.txt")

    # Mock duration (5 minutes)
    duration = 300.0

    with open(filename, 'w', encoding='utf-8') as f:
        _write_comprehensive_report(f, all_results, summaries, duration)

    print(f"Sample results file created: {filename}")
    print("\nThis demonstrates the format that will be used when running actual tests.")
    print("Run 'python testing/test_llm_models.py' to generate real test results.")


if __name__ == "__main__":
    main()
