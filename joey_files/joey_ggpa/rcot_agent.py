"""
rcot agent for minists game-playing.

implements the reflective chain-of-thought (rcot) approach from the paper
"language-driven play: large language models as game-playing agents in slay the spire"
by bateni & whitehead (fdg 2024).

key features:
- three prompt options: none (baseline), cot (chain-of-thought), rcot (reverse cot)
- card name anonymization for generalization
- saturn service discovery for zero-config ai access
- prompt structure: game context + game state + request
- designed for general game playing without specialized training
"""

from __future__ import annotations
import time
import random
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field

from ggpa.ggpa import GGPA
from ggpa.prompt2 import get_agent_target_prompt, get_card_target_prompt
from action.action import EndAgentTurn, PlayCard
from joey_setup.saturn_service_manager import SaturnServiceManager

if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card


@dataclass
class RCotConfig:
    """configuration for rcot agent."""
    model: str = "openrouter/auto"
    temperature: float = 0.7
    max_tokens: int = 500
    anonymize_cards: bool = True
    retry_limit: int = 3
    prompt_option: str = "cot"  # "none", "cot", or "rcot"


@dataclass
class RCotStatistics:
    """track agent performance metrics."""
    total_requests: int = 0
    invalid_responses: int = 0
    total_tokens: int = 0
    response_times: list[float] = field(default_factory=list)

    @property
    def avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    @property
    def invalid_rate(self) -> float:
        return (self.invalid_responses / self.total_requests * 100) if self.total_requests else 0.0


class RCotAgent(GGPA):
    """
    reflective chain-of-thought agent using llms for strategic game-playing.

    uses saturn service discovery to connect to ai services without api key management.
    implements the prompt structure from bateni & whitehead (2024).
    """

    def __init__(self, config: Optional[RCotConfig] = None,
                 saturn_manager: Optional[SaturnServiceManager] = None):
        self.config = config or RCotConfig()
        super().__init__(f"RCoT-{self.config.prompt_option}")

        # saturn service manager for zero-config ai discovery
        if saturn_manager is None:
            self.saturn_manager = SaturnServiceManager(discovery_timeout=3.0)
            self._owns_saturn = True
        else:
            self.saturn_manager = saturn_manager
            self._owns_saturn = False

        self.stats = RCotStatistics()
        self.card_anonymization_map = {}

    def _anonymize_card_name(self, card_name: str) -> str:
        """anonymize card names with random 6-character strings."""
        if not self.config.anonymize_cards:
            return card_name

        if card_name not in self.card_anonymization_map:
            # generate random 6-character string
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            self.card_anonymization_map[card_name] = ''.join(random.choice(chars) for _ in range(6))

        return self.card_anonymization_map[card_name]

    def _build_game_context(self, game_state: GameState, battle_state: BattleState) -> str:
        """
        build game context component of prompt.
        includes global game rules and card descriptions.
        """
        lines = ["=== GAME RULES ==="]
        lines.append("in this game, the player has a deck of cards.")
        lines.append("at the start of every turn, you draw cards from your draw pile.")
        lines.append("when the draw pile is empty, the discard pile is shuffled back into the draw pile.")
        lines.append(f"at the start of every turn, you gain {game_state.max_mana} mana.")
        lines.append("you can play cards by spending mana equal to the card's cost.")
        lines.append("after playing a card, it moves to the discard pile.")
        lines.append("when you end your turn, enemies perform their intended action.")
        lines.append("enemy attacks reduce your block first, then your hp.")
        lines.append("block is removed at the start of your turn.")
        lines.append("")
        lines.append("=== DECK COMPOSITION ===")

        # collect all unique cards in deck
        all_cards = battle_state.draw_pile + battle_state.discard_pile + battle_state.hand + battle_state.exhaust_pile
        card_counts = {}
        for card in all_cards:
            name = self._anonymize_card_name(card.name)
            if name not in card_counts:
                card_counts[name] = {'count': 0, 'card': card}
            card_counts[name]['count'] += 1

        for name, info in sorted(card_counts.items()):
            card = info['card']
            cost = card.mana_cost.peek()
            desc = card.get_description() if hasattr(card, 'get_description') else str(card)
            lines.append(f"{name} (cost {cost}): {desc} [{info['count']}x in deck]")

        return "\n".join(lines)

    def _build_game_state(self, game_state: GameState, battle_state: BattleState,
                         options: list[PlayCard | EndAgentTurn]) -> str:
        player = battle_state.player
        lines = [f"\n=== TURN {battle_state.turn} STATE ==="]
        lines.append(f"mana: {battle_state.mana}/{game_state.max_mana}")
        lines.append(f"player: {player.health}/{player.max_health} hp, {player.block} block")
        lines.append(f"status effects: {repr(player.status_effect_state)}")
        lines.append("")

        # enemy information with intentions
        lines.append("enemies:")
        for i, enemy in enumerate(battle_state.enemies):
            intent = enemy.get_intention(game_state, battle_state)
            lines.append(f"  {i}. {enemy.name}: {enemy.health}/{enemy.max_health} hp, {enemy.block} block")
            lines.append(f"     intent: {intent}")

        lines.append("")
        lines.append("=== YOUR OPTIONS ===")
        for i, option in enumerate(options):
            if isinstance(option, PlayCard):
                card = battle_state.hand[option.card_index]
                name = self._anonymize_card_name(card.name)
                cost = card.mana_cost.peek()
                desc = card.get_description() if hasattr(card, 'get_description') else str(card)
                lines.append(f"{i}. play {name} (cost {cost}): {desc}")
            else:
                lines.append(f"{i}. end turn")

        return "\n".join(lines)

    def _build_request(self, num_options: int) -> str:
        """
        prompt options from paper:
        - none: just ask for index
        - cot: ask for explanation first, then index
        - rcot: ask for index first, then explanation
        """
        lines = ["\n=== DECISION ==="]

        if self.config.prompt_option == "none":
            lines.append(f"respond with only the index (0-{num_options-1}) of the best option.")

        elif self.config.prompt_option == "cot":
            lines.append("in the first paragraph, explain which move you think is best and why.")
            lines.append(f"in the second paragraph, write only the index (0-{num_options-1}) of the best option.")

        elif self.config.prompt_option == "rcot":
            lines.append(f"in the first paragraph, write only the index (0-{num_options-1}) of the best option.")
            lines.append("in the second paragraph, explain why you think this move is best.")

        return "\n".join(lines)

    def _parse_response(self, content: str, max_index: int) -> Optional[int]:

        # split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [content.strip()]

        # for cot, index is in second paragraph; for rcot and none, it's in first
        target_paragraph = paragraphs[1] if self.config.prompt_option == "cot" and len(paragraphs) > 1 else paragraphs[0]

        # extract first number from target paragraph
        words = target_paragraph.replace('.', ' ').replace(',', ' ').split()
        for word in words:
            try:
                value = int(word)
                if 0 <= value < max_index:
                    return value
            except ValueError:
                continue

        return None

    def _make_api_call(self, prompt: str) -> Optional[dict]:
        """make api call via saturn service."""
        try:
            start = time.time()

            # single user message with full prompt
            messages = [{"role": "user", "content": prompt}]

            response = self.saturn_manager.chat_completion(
                messages=messages,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=30
            )

            elapsed = time.time() - start
            self.stats.total_requests += 1
            self.stats.response_times.append(elapsed)

            if response and 'usage' in response:
                self.stats.total_tokens += response['usage'].get('total_tokens', 0)

            return response

        except Exception as e:
            print(f"[rcot] api call failed: {e}")
            self.stats.invalid_responses += 1
            return None

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        """choose which card to play or end turn."""
        options = self.get_choose_card_options(game_state, battle_state)

        # build prompt with three components as per paper
        prompt_parts = []

        # only include full game context on turn 1 or every 5 turns to save tokens
        if battle_state.turn == 1 or battle_state.turn % 5 == 1:
            prompt_parts.append(self._build_game_context(game_state, battle_state))

        prompt_parts.append(self._build_game_state(game_state, battle_state, options))
        prompt_parts.append(self._build_request(len(options)))

        prompt = "\n".join(prompt_parts)

        # retry logic for invalid responses
        for attempt in range(self.config.retry_limit):
            response = self._make_api_call(prompt)

            if response is None:
                time.sleep(1)
                continue

            content = response['choices'][0]['message']['content'].strip()
            move_index = self._parse_response(content, len(options))

            if move_index is not None:
                return options[move_index]

            self.stats.invalid_responses += 1

        # fallback: random playable card or end turn
        playable = [opt for opt in options if isinstance(opt, PlayCard)]
        return random.choice(playable) if playable else options[-1]

    def choose_agent_target(self, battle_state: BattleState, list_name: str,
                          agent_list: list[Agent]) -> Agent:
        """choose which agent to target."""
        if len(agent_list) == 1:
            return agent_list[0]

        # simple heuristic: target lowest health enemy
        return min(agent_list, key=lambda a: a.health)

    def choose_card_target(self, battle_state: BattleState, list_name: str,
                          card_list: list[Card]) -> Card:
        """choose which card to target."""
        if len(card_list) == 1:
            return card_list[0]

        # simple heuristic: target first card
        return card_list[0]

    def get_statistics(self) -> dict:
        """get performance statistics."""
        return {
            'total_requests': self.stats.total_requests,
            'invalid_responses': self.stats.invalid_responses,
            'invalid_rate': self.stats.invalid_rate,
            'total_tokens': self.stats.total_tokens,
            'avg_response_time': self.stats.avg_response_time
        }

    def cleanup(self):
        """clean up saturn service manager."""
        if self._owns_saturn:
            self.saturn_manager.close()

    def __del__(self):
        """cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
