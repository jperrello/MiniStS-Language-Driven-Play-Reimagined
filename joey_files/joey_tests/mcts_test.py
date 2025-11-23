import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import JawWorm
from card import CardGen
import time
from ggpa.human_input import HumanInput
from ggpa.random_bot import RandomBot

try:
    from joey_ggpa.mcts_bot import MCTSAgent
except ImportError:
    print("Error: mcts_bot.py not found in ggpa/ directory")
    print("Make sure to copy mcts_bot.py to the ggpa/ folder")
    sys.exit(1)


def get_base_scenarios(scenario_name: str):
    scenarios = {
        'starter': (
            [
                CardGen.Strike(), CardGen.Strike(), CardGen.Strike(), CardGen.Strike(), CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Bash()
            ],
            20
        ),
        'basic': (
            [
                CardGen.Strike(), CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Bash(),
                CardGen.Cleave(), CardGen.Cleave(),
                CardGen.Anger()
            ],
            18
        ),
        'scaling': (
            [
                CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(),
                CardGen.SearingBlow(),
                CardGen.Armaments()
            ],
            16
        ),
        'vigor': (
            [
                CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Stimulate(), CardGen.Stimulate(),
                CardGen.Batter(), CardGen.Batter()
            ],
            15
        ),
        'lowhp': (
            [
                CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Bash(),
                CardGen.Impervious()
            ],
            8
        ),
        'bomb': (
            [
                CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Bomb(),
                CardGen.Bash()
            ],
            14
        ),
    }

    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        sys.exit(1)

    return scenarios[scenario_name]


def get_assignment_scenarios(scenario_name: str):
    scenarios = {
        'intro': (
            [
                CardGen.Strike(), CardGen.Strike(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Thunderclap(), CardGen.Thunderclap(), CardGen.Thunderclap(),
                CardGen.Inflame(),
                CardGen.Bash()
            ],
            20  # Target: 95%+ win rate (ideally 100%) with as few as 15 iterations
        ),
        'giant': (
            [
                CardGen.Strike(),
                CardGen.Bash(),
                CardGen.Defend(),
                CardGen.SearingBlow(),
                CardGen.Bludgeon()
            ],
            16  # Target: 95%+ win rate - strong offensive cards
        ),
        'offerings': (
            [
                CardGen.Strike(),
                CardGen.Offering(), CardGen.Offering(), CardGen.Offering(), CardGen.Offering(), CardGen.Offering(),
                CardGen.Thunderclap(), CardGen.Thunderclap(),
                CardGen.SearingBlow()
            ],
            19  # Target: 80-90% win rate - Offering costs 6 HP, only 3 safe uses
        ),
        'lowhp': (
            [
                CardGen.Strike(),
                CardGen.Offering(),
                CardGen.Defend(), CardGen.Defend(), CardGen.Defend(), CardGen.Defend(),
                CardGen.Thunderclap(), CardGen.Thunderclap(), CardGen.Thunderclap(), CardGen.Thunderclap()
            ],
            8  # Target: 65-80% win rate (low HP scenario - requires careful defense)
        ),
        'challenge': (
            [
                CardGen.Strike(),
                CardGen.Bash(),
                CardGen.Defend(),
                CardGen.SearingBlow(),
                CardGen.Bludgeon()
            ],
            8  # Target: Extremely difficult - requires 10000+ iterations, only ~3 turns max
        ),
    }

    if scenario_name not in scenarios:
        print(f"Unknown assignment scenario: {scenario_name}")
        print(f"Available assignment scenarios: {', '.join(scenarios.keys())}")
        sys.exit(1)

    return scenarios[scenario_name]


def run_game(bot, scenario_name: str, verbose: bool = False, is_assignment: bool = False) -> tuple:
    if is_assignment:
        deck, starting_hp = get_assignment_scenarios(scenario_name)
    else:
        deck, starting_hp = get_base_scenarios(scenario_name)

    game_state = GameState(Character.IRON_CLAD, bot, 0)
    game_state.set_deck(*deck)

    game_state.player.max_health = starting_hp
    game_state.player.health = starting_hp

    battle_state = BattleState(
        game_state,
        JawWorm(game_state),
        verbose=Verbose.LOG if verbose else Verbose.NO_LOG
    )

    initial_enemy_hp = sum(e.max_health for e in battle_state.enemies)

    start = time.time()
    battle_state.run()
    end = time.time()

    result = battle_state.get_end_result()
    won = result == 1

    # score calculation: win = 1.0, loss = damage dealt ratio
    if won:
        score = 1.0
    else:
        current_enemy_hp = sum(e.health for e in battle_state.enemies)
        damage_dealt = initial_enemy_hp - current_enemy_hp
        score = damage_dealt / initial_enemy_hp if initial_enemy_hp > 0 else 0

    time_taken = end - start

    return won, score, battle_state.turn, time_taken


def main():
    parser = argparse.ArgumentParser(description='Run MCTS bot on Slay the Spire scenarios')

    parser.add_argument('-n', '--iterations', type=int, default=50,
                       help='Number of MCTS iterations per turn (default: 50)')
    parser.add_argument('-s', '--scenario', type=str, default='starter',
                       help='Scenario to run: starter, basic, scaling, vigor, lowhp, bomb (default: starter)')
    parser.add_argument('-a', '--assignment', type=str, default=None,
                       help='Assignment scenario to run: intro, giant, offerings, lowhp, challenge, or "all" for all scenarios')
    parser.add_argument('-p', '--exploration', type=float, default=0.5,
                       help='Exploration parameter for UCB-1 (default: 0.5)')
    parser.add_argument('-g', '--games', type=int, default=1,
                       help='Number of games to run (default: 1)')
    parser.add_argument('-b', '--bot', type=str, default='mcts',
                       choices=['mcts', 'random', 'human'],
                       help='Bot to use (default: mcts)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    if args.bot == 'mcts':
        bot = MCTSAgent(iterations=args.iterations, exploration=args.exploration)
    elif args.bot == 'random':
        bot = RandomBot()
    elif args.bot == 'human':
        bot = HumanInput(True)
    else:
        print(f"Unknown bot: {args.bot}")
        sys.exit(1)

    # Determine if we're running assignment scenarios and which ones
    is_assignment = args.assignment is not None
    if is_assignment and args.assignment.lower() == 'all':
        scenarios_to_run = ['intro', 'giant', 'offerings', 'lowhp', 'challenge']
    elif is_assignment:
        scenarios_to_run = [args.assignment]
    else:
        scenarios_to_run = [args.scenario]

    # Target win rates for assignment scenarios
    assignment_targets = {
        'intro': 95.0,
        'giant': 95.0,
        'offerings': 85.0,  # 80-90% range, use midpoint
        'lowhp': 72.5,      # 65-80% range, use midpoint
        'challenge': None    # Extremely difficult, no clear target
    }

    # Run all scenarios
    for scenario_name in scenarios_to_run:
        if len(scenarios_to_run) > 1:
            print()
            print("=" * 60)
            print(f"RUNNING SCENARIO: {scenario_name.upper()}")
            print("=" * 60)

        print(f"Configuration:")
        if is_assignment:
            print(f"  Assignment Scenario: {scenario_name}")
            if scenario_name in assignment_targets and assignment_targets[scenario_name]:
                print(f"  Target Win Rate: {assignment_targets[scenario_name]:.1f}%")
        else:
            print(f"  Scenario: {scenario_name}")
        print(f"  Bot: {args.bot}")
        if args.bot == 'mcts':
            print(f"  Iterations: {args.iterations}")
            print(f"  Exploration (c): {args.exploration}")
        print(f"  Games: {args.games}")
        print()

        wins = 0
        total_score = 0
        total_turns = 0
        total_time = 0

        for game_num in range(args.games):
            verbose = args.verbose and args.games <= 3

            if args.games > 3 or verbose:
                if args.games > 1:
                    print(f"Game {game_num + 1}/{args.games}...", end=' ')

            won, score, turns, time_taken = run_game(bot, scenario_name, verbose, is_assignment)

            if won:
                wins += 1

            total_score += score
            total_turns += turns
            total_time += time_taken

            if args.games > 3:
                print(f"{'WIN' if won else 'LOSS'} (score: {score:.2f}, turns: {turns}, time: {time_taken:.2f}s)")
            elif args.games <= 3:
                print(f"\nResult: {'WIN' if won else 'LOSS'}")
                print(f"Score: {score:.2f}")
                print(f"Turns: {turns}")
                print(f"Time: {time_taken:.2f}s")
                print()

        # aggregate statistics for multiple game runs
        if args.games > 1:
            print()
            print("=" * 50)
            print("SUMMARY")
            print("=" * 50)
            print(f"Games played: {args.games}")
            win_rate = 100 * wins / args.games
            print(f"Wins: {wins} ({win_rate:.1f}%)")
            print(f"Losses: {args.games - wins} ({100 * (args.games - wins) / args.games:.1f}%)")

            # Show target comparison for assignment scenarios
            if is_assignment and scenario_name in assignment_targets:
                target = assignment_targets[scenario_name]
                if target:
                    if win_rate >= target:
                        status = "[TARGET MET]"
                    else:
                        status = "[BELOW TARGET]"
                    print(f"Target: {target:.1f}% - {status}")

            print(f"Average score: {total_score / args.games:.3f}")
            print(f"Average turns: {total_turns / args.games:.1f}")
            print(f"Average time: {total_time / args.games:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print()

        if args.verbose and args.bot == 'mcts' and args.games == 1:
            print()
            print("=" * 50)
            print("MCTS TREE (final turn)")
            print("=" * 50)
            bot.print_tree()


if __name__ == '__main__':
    main()