from __future__ import annotations
import re
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

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPNAI_API_KEY env var not set")
            
            self.client = OpenAI(api_key=api_key) 
            
        except Exception as e:
            print("\nAPI KEY ERROR")
            print(f"ERROR TYPE: {type(e)}")
            print(f"ERROR DETAILS: {e}")
            raise

        self.system_prompt = (
            "You are an expert card game strategist.\n"
            "Analyze the game state and enemy intent.\n"
            "You must choose ONLY ONE card from the list.\n"
            "You can only play the card if you have ENOUGH mana to play it.\n"
            "Your goal is to WIN the game and have as much health as you can remaining.\n"
            "Your response MUST end with a single line formatted EXACTLY as:\n"
            "CARD: <index>\n\n"
            "Ex:\n"
            "The enemy is attacking. I play Defend to block.\n"
            "CARD: 2"
        )

    def choose_agent_target(self, battle_state: BattleState, list_name: str, 
                            agent_list: list[Agent]) -> Agent:
        if len(agent_list) == 1:
            return agent_list[0]
        
        prompt = get_agent_target_prompt(battle_state, list_name, agent_list)
        
        response = self._api_call(prompt) 
        
        if response and response.choices:
            content = response.choices[0].message.content.strip()
            try:
                index = int(re.findall(r'\d+', content)[0])
                if 0 <= index < len(agent_list):
                    print(f"AGENT CHOSE TARGET: {index}")
                    return agent_list[index]
            except (ValueError, IndexError):
                print(f"ERROR IN CHOOSE AGENT TARGET")
                pass
        
        print("AGENT TARGET FALLBACK TO SMALLEST HP")
        return min(agent_list, key=lambda a: a.health)

    def choose_card_target(self, battle_state: BattleState, list_name: str, 
                           card_list: list[Card]) -> Card:
        if len(card_list) == 1:
            return card_list[0]
        
        prompt = get_card_target_prompt(battle_state, list_name, card_list)
        
        response = self._api_call(prompt)
        if response and response.choices:
            content = response.choices[0].message.content.strip()
            try:
                index = int(re.findall(r'\d+', content)[0])
                if 0 <= index < len(card_list):
                    print(f"CARD TARGET: {index}")
                    return card_list[index]
            except (ValueError, IndexError):
                print(f"ERROR IN CARD TARGET")
                pass

        return card_list[0]

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> Action:
            your_mana = battle_state.mana
            max_mana = game_state.max_mana

            enemy_info = ""
            for enemy in battle_state.enemies:
                enemy_info += f"\n{enemy.name}: HP {enemy.health}/{enemy.max_health}, Block {enemy.block}, Intent: {enemy.get_intention(game_state, battle_state)}"

            game_info = f"""
            You have {your_mana} <MANA> out of the {max_mana} <MANA> that you get every turn. Your HP: {battle_state.player.health}/{battle_state.player.max_health}
            Your Block: {battle_state.player.block}
            Your Status Effects: {repr(battle_state.player.status_effect_state)}
            Your Enemies are: {enemy_info}

            You have the following cards in your <EXHAUST_PILE>:
            {'-empty-' if len(battle_state.exhaust_pile) == 0 else
            ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.exhaust_pile)])}
            You have the following cards in your <DISCARD_PILE>:
            {'-empty-' if len(battle_state.discard_pile) == 0 else
            ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.discard_pile)])}
            You have the following cards in your <DRAW_PILE> (but in an unknown order):
            {'-empty-' if len(battle_state.draw_pile) == 0 else
            ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.draw_pile)])}
            You have the following cards in your <HAND>:
            {'-empty-' if len(battle_state.hand) == 0 else
            ' '.join([f'{i}: {card.get_name()}' for i, card in enumerate(battle_state.hand)])}
            """
            # cot_prompt = (
            #     "You are an expert card game strategist.\n"
            #     "Analyze the game state and enemy intent.\n"
            #     "You must choose ONLY ONE action from the list.\n"
            #     "Your response MUST end with a single line formatted EXACTLY as:\n"
            #     "Action: <index>\n\n"
            #     "Ex:\n"
            #     "The enemy is attacking. I play Defend to block.\n"
            #     "Action: 2"
            # )

            options: list[Action] = self.get_choose_card_options(game_state, battle_state)

            options_lines = []
            for i, action in enumerate(options):
                if isinstance(action, PlayCard):
                    card = battle_state.hand[action.card_index]
                    options_lines.append(f"{i}: Play {card.get_name()} (Cost: {card.mana_cost.peek()})")
                elif isinstance(action, EndAgentTurn):
                    options_lines.append(f"{i}: End Turn")
            
            your_options = " Your current options are:\n" + "\n".join(options_lines)

            #prompt = game_info + "\n" + cot_prompt + "\n" + your_options
            prompt = game_info + "\n" + your_options

            output = self._api_call(prompt)

            if not output.choices:
                print("\n PARSE ERROR: No choices in response. End Turn.")
                return options[-1]

            content = output.choices[0].message.content.strip()
            
            match = re.search(r"CARD:\s*(\d+)", content)
            
            action_index = None
            if match:
                try:
                    action_index = int(match.group(1))
                except ValueError:
                    print(f"Found 'Action:' but '{match.group(1)}' is not of int type.")
                    action_index = None
            
            # fall back
            if action_index is None:
                print("Could not find Action: <index> in response. FALLBACK to END TURN")
                return options[-1]
            
            if 0 <= action_index < len(options):
                action_chosen = options[action_index]
                print(f"AGENT CHOSE: {action_index}: {action_chosen.__class__.__name__}")
                return action_chosen
            else:
                print(f"PARSE ERROR: Index {action_index} is out of range (range: 0-{len(options)-1}). FALL BACK TO END TURN")
                return options[-1]
        
    def _api_call(self, prompt: str):
        try:
            #print(f"PROMPT: \n{prompt}\n")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )            
            return response

        except Exception as e:
            print(f"ERROR: {type(e)}, {e}")
            raise