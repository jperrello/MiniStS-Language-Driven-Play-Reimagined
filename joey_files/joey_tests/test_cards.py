#!/usr/bin/env python3
"""
Unified Card Testing Suite for Slay the Spire LLM Project

Test custom cards against different AI agents (LLM, MCTS, RCoT, Backtrack, Random).

Usage Examples:
    # Test with defaults
    python joey_tests/test_cards.py

    # Test custom cards against LLM agent using GPT-4o
    python joey_tests/test_cards.py --agent llm --model gpt-4o --cards my_cards.json --enemy jj --games 10

    # Test multiple agents against starter deck
    python joey_tests/test_cards.py --agent mcts,llm,random --scenario starter --enemy j --games 5

    # Quick comparison of all agents
    python joey_tests/test_cards.py --agent all --scenario basic --games 3
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import JawWorm, Goblin, HobGoblin, Leech, AcidSlimeSmall, SpikeSlimeSmall
from card import CardGen

from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from joey_files.joey_ggpa.mcts_bot import MCTSAgent
from joey_files.joey_ggpa.llm_bot import LLMBot
from joey_files.joey_ggpa.rcot_agent import RCotAgent, RCotConfig
from ggpa.prompt2 import PromptOption
from joey_files.joey_setup.saturn_service_manager import SaturnServiceManager


@dataclass
class GameResult:
    won: bool
    turns: int
    time_taken: float
    damage_dealt: float
    player_hp_remaining: int


@dataclass
class AgentStats:
    agent_name: str
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    total_turns: int = 0
    total_time: float = 0.0
    total_damage: float = 0.0
    turn_counts: List[int] = field(default_factory=list)
    time_per_game: List[float] = field(default_factory=list)
    damage_percentages: List[float] = field(default_factory=list)
    hp_remaining: List[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_games * 100) if self.total_games > 0 else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.total_games if self.total_games > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_games if self.total_games > 0 else 0.0

    @property
    def avg_damage(self) -> float:
        return self.total_damage / self.total_games if self.total_games > 0 else 0.0

    @property
    def stdev_turns(self) -> float:
        return stdev(self.turn_counts) if len(self.turn_counts) > 1 else 0.0

    @property
    def stdev_time(self) -> float:
        return stdev(self.time_per_game) if len(self.time_per_game) > 1 else 0.0

    def add_result(self, result: GameResult):
        self.total_games += 1
        if result.won:
            self.wins += 1
        else:
            self.losses += 1
        self.total_turns += result.turns
        self.total_time += result.time_taken
        self.total_damage += result.damage_dealt
        self.turn_counts.append(result.turns)
        self.time_per_game.append(result.time_taken)
        self.damage_percentages.append(result.damage_dealt)
        self.hp_remaining.append(result.player_hp_remaining)


SCENARIOS = {
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
}

ENEMY_CONFIGS = {
    'j': [JawWorm],
    'jj': [JawWorm, JawWorm],
    'g': [Goblin],
    'gg': [Goblin, Goblin],
    'jg': [JawWorm, Goblin],
    'jgg': [JawWorm, Goblin, Goblin],
    'h': [HobGoblin],
    'l': [Leech],
    'as': [AcidSlimeSmall],
    'ss': [SpikeSlimeSmall],
}


def load_custom_cards(cards_path: str) -> Tuple[List[Any], int]:
    """THIS FUNCTION IS WHERE WE LOAD CUSTOM CARDS FROM A JSON FILE."""
    with open(cards_path, 'r') as f:
        data = json.load(f)

    cards = []
    for card_name in data['cards']:
        if hasattr(CardGen, card_name):
            cards.append(getattr(CardGen, card_name)())
        else:
            print(f"Warning: Unknown card '{card_name}' - skipping")

    starting_hp = data.get('starting_hp', 20)
    return cards, starting_hp


def get_deck_and_hp(args) -> Tuple[List[Any], int]:
    if args.cards:
        return load_custom_cards(args.cards)
    elif args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Error: Unknown scenario '{args.scenario}'")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        return SCENARIOS[args.scenario]
    else:
        return SCENARIOS['starter']


def create_enemies(enemy_config: str, game_state: GameState) -> List[Any]:
    if enemy_config not in ENEMY_CONFIGS:
        print(f"Error: Unknown enemy config '{enemy_config}'")
        print(f"Available: {', '.join(ENEMY_CONFIGS.keys())}")
        sys.exit(1)

    return [enemy_class(game_state) for enemy_class in ENEMY_CONFIGS[enemy_config]]


def create_agent(agent_name: str, model_name: Optional[str], saturn_manager: Optional[SaturnServiceManager]):
    agent_name = agent_name.lower()

    if agent_name == 'random':
        return RandomBot()

    elif agent_name == 'backtrack':
        return BacktrackBot(depth=4, should_save_states=False)

    elif agent_name == 'mcts':
        return MCTSAgent(iterations=100, exploration=0.5)

    elif agent_name == 'llm':
        if not model_name:
            model_name = 'gpt-4o-mini'
        return LLMBot(
            model_name=model_name,
            prompt_option=PromptOption.CoT,
            few_shot=0,
            show_option_results=False,
            use_structured_output=False,
            saturn_manager=saturn_manager
        )

    elif agent_name == 'rcot':
        if not model_name:
            model_name = 'gpt-4o'
        return RCotAgent(
            config=RCotConfig(model=model_name, prompt_option='cot'),
            saturn_manager=saturn_manager
        )

    else:
        print(f"Error: Unknown agent '{agent_name}'")
        print("Available: random, backtrack, mcts, llm, rcot, all")
        sys.exit(1)


def run_single_game(agent, deck: List[Any], starting_hp: int, enemy_config: str, verbose: bool) -> GameResult:
    game_state = GameState(Character.IRON_CLAD, agent, 0)
    game_state.set_deck(*deck)
    game_state.player.max_health = starting_hp
    game_state.player.health = starting_hp

    enemies = create_enemies(enemy_config, game_state)
    battle_state = BattleState(
        game_state,
        *enemies,
        verbose=Verbose.LOG if verbose else Verbose.NO_LOG
    )

    initial_enemy_hp = sum(e.max_health for e in battle_state.enemies)

    start_time = time.time()
    battle_state.run()
    end_time = time.time()

    result = battle_state.get_end_result()
    won = result == 1

    current_enemy_hp = sum(e.health for e in battle_state.enemies)
    damage_dealt = (initial_enemy_hp - current_enemy_hp) / initial_enemy_hp if initial_enemy_hp > 0 else 0.0

    return GameResult(
        won=won,
        turns=battle_state.turn,
        time_taken=end_time - start_time,
        damage_dealt=damage_dealt,
        player_hp_remaining=battle_state.player.health
    )


def print_progress(current: int, total: int, agent_name: str):

    progress = current / total
    filled = int(50 * progress)
    bar = "=" * filled + "-" * (50 - filled)
    percentage = progress * 100
    print(f"\r  {agent_name:20s} [{bar}] {current}/{total} ({percentage:.1f}%)", end='', flush=True)


def print_results_table(all_stats: List[AgentStats]):
    print("\n" + "=" * 100)
    print("TEST RESULTS SUMMARY")
    print("=" * 100)

    sorted_stats = sorted(all_stats, key=lambda s: s.win_rate, reverse=True)

    print(f"\n{'Agent':<20} {'Games':>6} {'Wins':>6} {'Win %':>8} {'Avg Turns':>10} {'Avg Time':>10} {'Avg Dmg %':>10}")
    print("-" * 100)

    for stats in sorted_stats:
        print(f"{stats.agent_name:<20} "
              f"{stats.total_games:>6} "
              f"{stats.wins:>6} "
              f"{stats.win_rate:>7.2f}% "
              f"{stats.avg_turns:>10.2f} "
              f"{stats.avg_time:>9.3f}s "
              f"{stats.avg_damage * 100:>9.2f}%")

    print()


def save_results(all_stats: List[AgentStats], output_path: str):
    """saved to JSON file."""
    results = {
        'agents': [
            {
                'name': s.agent_name,
                'games': s.total_games,
                'wins': s.wins,
                'losses': s.losses,
                'win_rate': s.win_rate,
                'avg_turns': s.avg_turns,
                'avg_time': s.avg_time,
                'avg_damage': s.avg_damage,
            }
            for s in all_stats
        ]
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test custom cards against different AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with defaults (llm agent, starter deck, single JawWorm, 5 games)
  python joey_tests/test_cards.py

  # Test custom cards
  python joey_tests/test_cards.py --cards my_cards.json --enemy jj --games 10

  # Test multiple agents
  python joey_tests/test_cards.py --agent mcts,llm,random --scenario starter --games 5

  # Compare all agents
  python joey_tests/test_cards.py --agent all --scenario basic --games 3

Enemy Configurations:
  j=JawWorm, jj=2xJawWorm, g=Goblin, gg=2xGoblin, jg=JawWorm+Goblin, etc.

Scenarios:
  starter, basic, scaling, vigor, lowhp
        """
    )

    parser.add_argument('--agent', type=str, default='llm',
                       help='Agent(s) to test (comma-separated or "all"). Options: llm, mcts, rcot, backtrack, random')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model to use (for LLM/RCoT agents). Default: gpt-4o-mini for LLM, gpt-4o for RCoT')
    parser.add_argument('--cards', type=str, default=None,
                       help='Path to custom cards JSON file')
    parser.add_argument('--scenario', type=str, default='starter',
                       help='Predefined scenario to test (default: starter)')
    parser.add_argument('--enemy', type=str, default='j',
                       help='Enemy configuration (default: j)')
    parser.add_argument('--games', type=int, default=5,
                       help='Number of games to run (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed game logs')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON (optional)')

    args = parser.parse_args()

    # Parse agent list
    if args.agent.lower() == 'all':
        agent_names = ['random', 'backtrack', 'mcts', 'llm', 'rcot']
    else:
        agent_names = [a.strip() for a in args.agent.split(',')]

    # Get deck configuration
    deck, starting_hp = get_deck_and_hp(args)

    print("=" * 100)
    print("CARD TESTING SUITE")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Deck:    {'Custom cards' if args.cards else args.scenario} ({len(deck)} cards, {starting_hp} HP)")
    print(f"  Enemy:   {args.enemy}")
    print(f"  Agents:  {', '.join(agent_names)}")
    print(f"  Games:   {args.games} per agent")
    print(f"  Verbose: {args.verbose}")
    print()

    # Initialize SATURN manager if needed
    saturn_manager = None
    if any(a in agent_names for a in ['llm', 'rcot']):
        print("Initializing SATURN service discovery...")
        try:
            saturn_manager = SaturnServiceManager(discovery_timeout=3.0)
        except Exception as e:
            print(f"Error: Failed to initialize SATURN: {e}")
            print("Make sure a SATURN server is running: python joey_setup/saturn_server.py")
            sys.exit(1)

    all_stats = []

    try:
        for agent_idx, agent_name in enumerate(agent_names, 1):
            print(f"\n[{agent_idx}/{len(agent_names)}] Testing {agent_name}...")

            agent = create_agent(agent_name, args.model, saturn_manager)
            stats = AgentStats(agent_name=agent_name)

            for game_num in range(args.games):
                print_progress(game_num + 1, args.games, agent_name)

                verbose = args.verbose and game_num == 0

                try:
                    result = run_single_game(agent, deck, starting_hp, args.enemy, verbose)
                    stats.add_result(result)
                except Exception as e:
                    print(f"\n  Error in game {game_num + 1}: {e}")
                    continue

            print()
            print(f"  Completed: {stats.wins}/{args.games} wins ({stats.win_rate:.1f}%), "
                  f"avg {stats.avg_turns:.1f} turns, {stats.avg_time:.2f}s per game")

            all_stats.append(stats)

    finally:
        if saturn_manager:
            print("\nCleaning up SATURN service manager...")
            saturn_manager.close()

    # Print results
    print_results_table(all_stats)

    # Save results if requested
    if args.output:
        save_results(all_stats, args.output)

    # Print best performer
    if all_stats:
        best = max(all_stats, key=lambda s: s.win_rate)
        print(f"Best performing agent: {best.agent_name} with {best.win_rate:.2f}% win rate\n")


if __name__ == '__main__':
    main()
