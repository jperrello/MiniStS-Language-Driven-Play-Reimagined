#!/usr/bin/env python3

import argparse
import time
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import JawWorm, Goblin, HobGoblin, Leech, AcidSlimeSmall, SpikeSlimeSmall
from card import CardGen

from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.mcts_bot import MCTSAgent
from ggpa.chatgpt_bot import ChatGPTBot
from ggpa.prompt2 import PromptOption
from rcot_agent import RCotAgent, RCotConfig


@dataclass
class AgentConfig:
    name: str
    agent_instance: Any
    description: str


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
    def loss_rate(self) -> float:
        return (self.losses / self.total_games * 100) if self.total_games > 0 else 0.0

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


def get_scenario_deck(scenario_name: str) -> Tuple[List[Any], int]:
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
                CardGen.Searing_Blow(),
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

    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        sys.exit(1)

    return scenarios[scenario_name]


def get_enemy(enemy_name: str, game_state: GameState):
    enemies = {
        'jawworm': JawWorm,
        'goblin': Goblin,
        'hobgoblin': HobGoblin,
        'leech': Leech,
        'acidslime': AcidSlimeSmall,
        'spikeslime': SpikeSlimeSmall,
    }

    enemy_name = enemy_name.lower()
    if enemy_name not in enemies:
        print(f"Unknown enemy: {enemy_name}")
        print(f"Available enemies: {', '.join(enemies.keys())}")
        sys.exit(1)

    return enemies[enemy_name](game_state)


def run_single_game(agent, scenario_name: str, enemy_name: str, verbose: bool = False) -> GameResult:
    deck, starting_hp = get_scenario_deck(scenario_name)

    game_state = GameState(Character.IRON_CLAD, agent, 0)
    game_state.set_deck(*deck)

    game_state.player.max_health = starting_hp
    game_state.player.health = starting_hp

    enemy = get_enemy(enemy_name, game_state)
    battle_state = BattleState(
        game_state,
        enemy,
        verbose=Verbose.LOG if verbose else Verbose.NO_LOG
    )

    initial_enemy_hp = sum(e.max_health for e in battle_state.enemies)

    start_time = time.time()
    battle_state.run()
    end_time = time.time()

    result = battle_state.get_end_result()
    won = result == 1

    # calculate damage as fraction of initial enemy hp
    current_enemy_hp = sum(e.health for e in battle_state.enemies)
    damage_dealt = (initial_enemy_hp - current_enemy_hp) / initial_enemy_hp if initial_enemy_hp > 0 else 0.0

    player_hp_remaining = battle_state.player.health

    return GameResult(
        won=won,
        turns=battle_state.turn,
        time_taken=end_time - start_time,
        damage_dealt=damage_dealt,
        player_hp_remaining=player_hp_remaining
    )


def get_all_agents() -> List[AgentConfig]:
    agents = [
        AgentConfig(
            name="Random",
            agent_instance=RandomBot(),
            description="Random action selection (baseline)"
        ),

        AgentConfig(
            name="Backtrack-D3",
            agent_instance=BacktrackBot(depth=3, should_save_states=False),
            description="Minimax search with depth 3"
        ),
        AgentConfig(
            name="Backtrack-D4",
            agent_instance=BacktrackBot(depth=4, should_save_states=False),
            description="Minimax search with depth 4"
        ),
        AgentConfig(
            name="Backtrack-D5",
            agent_instance=BacktrackBot(depth=5, should_save_states=False),
            description="Minimax search with depth 5"
        ),

        AgentConfig(
            name="MCTS-50",
            agent_instance=MCTSAgent(iterations=50, exploration=0.5),
            description="MCTS with 50 iterations, c=0.5"
        ),
        AgentConfig(
            name="MCTS-100",
            agent_instance=MCTSAgent(iterations=100, exploration=0.5),
            description="MCTS with 100 iterations, c=0.5"
        ),
        AgentConfig(
            name="MCTS-200",
            agent_instance=MCTSAgent(iterations=200, exploration=0.5),
            description="MCTS with 200 iterations, c=0.5"
        ),

        AgentConfig(
            name="MCTS-100-c1.0",
            agent_instance=MCTSAgent(iterations=100, exploration=1.0),
            description="MCTS with 100 iterations, c=1.0 (more exploration)"
        ),
        AgentConfig(
            name="MCTS-100-c0.2",
            agent_instance=MCTSAgent(iterations=100, exploration=0.2),
            description="MCTS with 100 iterations, c=0.2 (more exploitation)"
        ),

        # llm-based agents require OPENAI_API_KEY
        AgentConfig(
            name="RCoT-GPT4o",
            agent_instance=RCotAgent(RCotConfig(model="gpt-4o", temperature=0.7, max_tokens=500)),
            description="GPT-4o with Reverse CoT reasoning (temp=0.7)"
        ),
        AgentConfig(
            name="RCoT-GPT4o-T0.3",
            agent_instance=RCotAgent(RCotConfig(model="gpt-4o", temperature=0.3, max_tokens=500)),
            description="GPT-4o with Reverse CoT reasoning (temp=0.3, more focused)"
        ),
        AgentConfig(
            name="RCoT-GPT4o-T1.0",
            agent_instance=RCotAgent(RCotConfig(model="gpt-4o", temperature=1.0, max_tokens=500)),
            description="GPT-4o with Reverse CoT reasoning (temp=1.0, more creative)"
        ),

        AgentConfig(
            name="GPT4-CoT",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_4,
                prompt_option=PromptOption.CoT,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-4 with Chain of Thought prompting"
        ),
        AgentConfig(
            name="GPT4-RCoT",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_4,
                prompt_option=PromptOption.CoT_rev,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-4 with Reverse Chain of Thought prompting"
        ),
        AgentConfig(
            name="GPT4-DAG",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_4,
                prompt_option=PromptOption.DAG,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-4 with DAG (Directed Acyclic Graph) prompting"
        ),
        AgentConfig(
            name="GPT4T-CoT",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_Turbo_4,
                prompt_option=PromptOption.CoT,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-4-Turbo with Chain of Thought prompting"
        ),
        AgentConfig(
            name="GPT3.5T-CoT",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_Turbo_35,
                prompt_option=PromptOption.CoT,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-3.5-Turbo with Chain of Thought prompting"
        ),
        AgentConfig(
            name="GPT3.5T-RCoT",
            agent_instance=ChatGPTBot(
                model_name=ChatGPTBot.ModelName.GPT_Turbo_35,
                prompt_option=PromptOption.CoT_rev,
                few_shot=0,
                show_option_results=False
            ),
            description="GPT-3.5-Turbo with Reverse Chain of Thought prompting"
        ),
    ]

    return agents


def print_header(title: str):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_progress_bar(current: int, total: int, agent_name: str, width: int = 50):
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress * 100
    print(f"\r  {agent_name:20s} [{bar}] {current}/{total} ({percentage:.1f}%)", end='', flush=True)


def print_results_table(all_stats: List[AgentStats]):
    print_header("TEST RESULTS SUMMARY")

    # sort by win rate descending for ranking
    sorted_stats = sorted(all_stats, key=lambda s: s.win_rate, reverse=True)

    print()
    print(f"{'Agent':<20} {'Games':>6} {'Wins':>6} {'Losses':>6} {'Win %':>8} {'Loss %':>8} "
          f"{'Avg Turns':>10} {'Avg Time':>10} {'Avg Dmg %':>10}")
    print("-" * 100)

    for stats in sorted_stats:
        print(f"{stats.agent_name:<20} "
              f"{stats.total_games:>6} "
              f"{stats.wins:>6} "
              f"{stats.losses:>6} "
              f"{stats.win_rate:>7.2f}% "
              f"{stats.loss_rate:>7.2f}% "
              f"{stats.avg_turns:>10.2f} "
              f"{stats.avg_time:>9.3f}s "
              f"{stats.avg_damage * 100:>9.2f}%")

    print()


def print_detailed_stats(all_stats: List[AgentStats]):
    print_header("DETAILED STATISTICS")

    sorted_stats = sorted(all_stats, key=lambda s: s.win_rate, reverse=True)

    for stats in sorted_stats:
        print()
        print(f"Agent: {stats.agent_name}")
        print(f"  Games Played:     {stats.total_games}")
        print(f"  Wins:             {stats.wins} ({stats.win_rate:.2f}%)")
        print(f"  Losses:           {stats.losses} ({stats.loss_rate:.2f}%)")
        print(f"  Average Turns:    {stats.avg_turns:.2f} ± {stats.stdev_turns:.2f}")
        print(f"  Average Time:     {stats.avg_time:.3f}s ± {stats.stdev_time:.3f}s")
        print(f"  Total Time:       {stats.total_time:.2f}s")
        print(f"  Average Damage:   {stats.avg_damage * 100:.2f}%")
        if stats.wins > 0:
            winning_games = [stats.hp_remaining[i] for i in range(len(stats.hp_remaining))
                           if stats.damage_percentages[i] >= 1.0]
            if winning_games:
                print(f"  Avg HP Remaining: {mean(winning_games):.2f} (winning games only)")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Test all agents in the codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_all_agents.py                      # Run with defaults
  python test_all_agents.py --games 100          # Run 100 games per agent
  python test_all_agents.py --scenario basic     # Test on basic scenario
  python test_all_agents.py --enemy goblin       # Test against goblin
  python test_all_agents.py --quick              # Quick test (10 games, fewer agents)
        """
    )

    parser.add_argument('-g', '--games', type=int, default=25,
                       help='Number of games to run per agent (default: 25)')
    parser.add_argument('-s', '--scenario', type=str, default='starter',
                       choices=['starter', 'basic', 'scaling', 'vigor', 'lowhp'],
                       help='Scenario to test (default: starter)')
    parser.add_argument('-e', '--enemy', type=str, default='jawworm',
                       choices=['jawworm', 'goblin', 'hobgoblin', 'leech', 'acidslime', 'spikeslime'],
                       help='Enemy to fight (default: jawworm)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose game output (only for first game of each agent)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (10 games, fewer agents)')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed statistics')

    args = parser.parse_args()

    if args.quick:
        args.games = 10
        print("\n⚡ Quick test mode: 10 games per agent, testing core agents only\n")

    print_header("AGENT TESTING SUITE - MiniStS")
    print()
    print(f"Configuration:")
    print(f"  Scenario:         {args.scenario}")
    print(f"  Enemy:            {args.enemy}")
    print(f"  Games per agent:  {args.games}")
    print(f"  Verbose:          {args.verbose}")
    print()

    all_agents = get_all_agents()

    if args.quick:
        all_agents = [a for a in all_agents if a.name in
                     ['Random', 'Backtrack-D4', 'MCTS-50', 'MCTS-100']]

    print(f"Testing {len(all_agents)} agents:")
    for agent_config in all_agents:
        print(f"  • {agent_config.name:<20} - {agent_config.description}")
    print()

    all_stats: List[AgentStats] = []
    total_start_time = time.time()

    for agent_idx, agent_config in enumerate(all_agents, 1):
        print(f"\n[{agent_idx}/{len(all_agents)}] Testing {agent_config.name}...")

        stats = AgentStats(agent_name=agent_config.name)

        for game_num in range(args.games):
            print_progress_bar(game_num + 1, args.games, agent_config.name)

            verbose = args.verbose and game_num == 0
            try:
                result = run_single_game(
                    agent_config.agent_instance,
                    args.scenario,
                    args.enemy,
                    verbose
                )
                stats.add_result(result)
            except Exception as e:
                print(f"\n  Error in game {game_num + 1}: {e}")
                continue

        print()
        print(f"  ✓ Completed: {stats.wins}/{args.games} wins ({stats.win_rate:.1f}%), "
              f"avg {stats.avg_turns:.1f} turns, {stats.avg_time:.2f}s per game")

        all_stats.append(stats)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print_results_table(all_stats)

    if args.detailed:
        print_detailed_stats(all_stats)

    print_header("OVERALL SUMMARY")
    print()
    print(f"Total agents tested:  {len(all_agents)}")
    print(f"Games per agent:      {args.games}")
    print(f"Total games played:   {len(all_agents) * args.games}")
    print(f"Total time:           {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print(f"Average per agent:    {total_time / len(all_agents):.2f}s")
    print()

    best_agent = max(all_stats, key=lambda s: s.win_rate)
    print(f"🏆 Best performing agent: {best_agent.agent_name} with {best_agent.win_rate:.2f}% win rate")
    print()


if __name__ == '__main__':
    main()