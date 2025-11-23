from __future__ import annotations
from openai import OpenAI
import time
import json
import os
from enum import StrEnum
from ggpa.ggpa import GGPA
from action.action import EndAgentTurn, PlayCard
from auth import GPT_AUTH
from utility import get_unique_filename
from ggpa.prompt2 import PromptOption, get_action_prompt,\
    get_agent_target_prompt, get_card_target_prompt,\
    strip_response, _get_game_context
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card
    from action.action import Action

class CotAgent(GGPA):
    API_KEY = GPT_AUTH
    tokens = 500

    def __init__(self) -> None:
        super().__init__(name="CotAgent")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """
        During your turn you can draw a single card from your draw pile, or play a card from your hand.
        You can play a card if you have enough mana to play it
        """

    def _get_action(self, game_state: GameState, battle_state: BattleState) -> Action:
        # TODO chatgpt chooses next card action based on the prompt & current game state/battle state

        deckset = set()
        deckset.add(battle_state.exhaust_pile + battle_state.discard_pile + battle_state.draw_pile + battle_state.hand)
        deck = list(deckset)

        your_mana = battle_state.mana
        max_mana = game_state.max_mana

        enemy_info = ""
        for enemy in battle_state.enemies:
            enemy_info += f"\n{enemy.name}: HP {enemy.health}/{enemy.max_health}, Block {enemy.block}, Intent: {enemy.get_intention(game_state, battle_state)}"

        game_info = f"""
        You have {your_mana} <MANA> out of the {max_mana} <MANA> that you get every turn.
        Your HP: {battle_state.player.health}/{battle_state.player.max_health}
        Your Block: {battle_state.player.block}
        Your Status Effects: {repr(battle_state.player.status_effect_state)}
        Your Enemies are: {enemy_info}

        You have the following cards in your <EXHAUST_PILE>:
        {'-empty-' if len(battle_state.exhaust_pile) == 0 else
         ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.exhaust_pile)])}
        You have the following cards in your <DISCARD_PILE>:
        {'-empty-' if len(battle_state.discard_pile) == 0 else
         ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.discard_pile)])}
        You have the following cards in your <DRAW_PILE>, but in an unknown order:
        {'-empty-' if len(battle_state.draw_pile) == 0 else
         ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.draw_pile)])}
        You have the following cards in your <HAND>:
        {'-empty-' if len(battle_state.hand) == 0 else
         ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.hand)])}
        """

        """
        prompt components
        
Global info about the game
a. Info about drawing cards
b. Spending mana
c. Status effects
d. Description of existing cards in the deck
2. Game state
a. Mana
b. Turn number
c. Hp
d. Block and status effects of player
e. Block and status effects of enemy
f. Intention of enemy
"""

        cot_prompt = """Explain the best move given the below scenario. A few lines below your explanation, give the index of the best move."""

        options = battle_state.get_hand()
        your_options = """ Your current options are: {}""".format(", ".join([str(x) for x in options]))


        prompt = cot_prompt + game_info + your_options

        output = self._api_call(prompt)

        content = output.choices[0].message.content.strip()

        action_index = int(content.split(" ")[-1])

        return options[action_index]

    def _api_call(self, prompt: str):

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return response