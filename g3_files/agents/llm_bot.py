from __future__ import annotations
import time
import os
from enum import StrEnum
from openai import OpenAI
from ggpa.ggpa import GGPA
from action.action import EndAgentTurn, PlayCard
from utility import get_unique_filename
from ggpa.prompt2 import PromptOption, get_action_prompt,\
    get_agent_target_prompt, get_card_target_prompt,\
    strip_response
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card
    from action.action import Action


class SimpleLLMBot(GGPA):

    class ModelName(StrEnum):
        OPENROUTER_AUTO = "openrouter/auto"
        GPT_4o = "openai/gpt-4o"
        CLAUDE_35_SONNET = "anthropic/claude-3.5-sonnet"
        GEMINI_PRO_15 = "google/gemini-2.5-pro"
        GROK_41_FAST_FREE = "x-ai/grok-4.1-fast:free"
        GEMMA_3N_FREE = "google/gemma-3n-e2b-it:free"
        NEMOTRON_NANO_FREE = "nvidia/nemotron-nano-9b-v2:free"
        GPT_OSS_20B_FREE = "openai/gpt-oss-20b:free"
        DEEPSEEK_R1T2_FREE = "tngtech/deepseek-r1t2-chimera:free"

    def __init__(
        self,
        model_name: SimpleLLMBot.ModelName,
        prompt_option: PromptOption,
        few_shot: int,
        show_option_results: bool,
    ):
        model_name_dict = {
            SimpleLLMBot.ModelName.OPENROUTER_AUTO: 'or-auto',
            SimpleLLMBot.ModelName.GPT_4o: '4o',
            SimpleLLMBot.ModelName.CLAUDE_35_SONNET: 'claude',
            SimpleLLMBot.ModelName.GEMINI_PRO_15: 'gemini',
            SimpleLLMBot.ModelName.GROK_41_FAST_FREE: 'grok-free',
            SimpleLLMBot.ModelName.GEMMA_3N_FREE: 'gemma-free',
            SimpleLLMBot.ModelName.NEMOTRON_NANO_FREE: 'nemotron-free',
            SimpleLLMBot.ModelName.GPT_OSS_20B_FREE: 'gpt-oss-free',
            SimpleLLMBot.ModelName.DEEPSEEK_R1T2_FREE: 'deepseek-free',
        }
        prompt_dict = {
            PromptOption.NONE: 'none',
            PromptOption.CoT: 'cot',
            PromptOption.CoT_rev: 'cotr',
            PromptOption.DAG: 'dag',
        }
        self.model_name = model_name
        self.prompt_option = prompt_option
        self.few_shot = few_shot
        self.show_option_results = show_option_results
        self.messages: list[dict[str, str]] = []
        self._client: Optional[OpenAI] = None

        super().__init__(f"LLM-{model_name_dict[model_name]}-{prompt_dict[prompt_option]}-f{self.few_shot}{'-results' if show_option_results else ''}")
        self.clear_metadata()
        self.clear_history()

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._client

    def ask_llm(self) -> str:
        try:
            before_request = time.time()

            response = self.client.chat.completions.create(
                model=str(self.model_name),
                messages=self.messages,
                max_tokens=500,
                temperature=0.7,
            )

            elapsed = time.time() - before_request
            self.metadata["response_time"].append(elapsed)
            print(f"[LLM] Response time: {elapsed:.2f}s")

            return response.choices[0].message.content

        except Exception as e:
            print(f"[LLM] Error making OpenRouter request: {e}")
            raise

    def format_prompt(self, options: list[str], option_objects: list, game_state: GameState, battle_state: BattleState) -> str:
        prompt_content = get_action_prompt(
            game_state, battle_state, option_objects,
            self.prompt_option, True, self.show_option_results
        )
        return prompt_content

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        options: list[PlayCard | EndAgentTurn] = self.get_choose_card_options(game_state, battle_state)

        if len(options) == 1:
            return options[0]

        option_strings = [str(opt) for opt in options]
        prompt_content = self.format_prompt(option_strings, options, game_state, battle_state)

        self.messages = [
            {"role": "system", "content": "You are a Slay the Spire game-playing AI. Respond ONLY with the number of your chosen action."},
            {"role": "user", "content": prompt_content}
        ]

        response = self.ask_llm()

        extracted = strip_response(response, self.prompt_option)

        if not extracted or not extracted.isdigit():
            print(f"Wrong format for {self.name}, option {self.prompt_option}:")
            print(response)
            print(f"Extracted: {extracted}")
            return options[0]

        try:
            index = int(extracted)
            if 0 <= index < len(options):
                return options[index]
        except:
            pass

        return options[0]

    def choose_agent_target(self, battle_state: BattleState, list_name: str, targets: list[Agent]) -> Agent:
        if len(targets) <= 1:
            return targets[0]

        prompt_content = get_agent_target_prompt(battle_state, list_name, targets)

        self.messages = [
            {"role": "system", "content": "You are a Slay the Spire game-playing AI. Respond ONLY with the number of your chosen target."},
            {"role": "user", "content": prompt_content}
        ]

        response = self.ask_llm()
        extracted = strip_response(response, self.prompt_option)

        if not extracted or not extracted.isdigit():
            return targets[0]

        try:
            index = int(extracted)
            if 0 <= index < len(targets):
                return targets[index]
        except:
            pass

        return targets[0]

    def choose_card_target(self, battle_state: BattleState, list_name: str, cards: list[Card]) -> Card:
        if len(cards) <= 1:
            return cards[0]

        prompt_content = get_card_target_prompt(battle_state, list_name, cards)

        self.messages = [
            {"role": "system", "content": "You are a Slay the Spire game-playing AI. Respond ONLY with the number of your chosen card."},
            {"role": "user", "content": prompt_content}
        ]

        response = self.ask_llm()
        extracted = strip_response(response, self.prompt_option)

        if not extracted or not extracted.isdigit():
            return cards[0]

        try:
            index = int(extracted)
            if 0 <= index < len(cards):
                return cards[index]
        except:
            pass

        return cards[0]

    def clear_history(self):
        self.messages = []

    def clear_metadata(self):
        self.metadata = {
            "response_time": [],
            "wrong_format": 0,
        }
