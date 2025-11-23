"""
LLM Bot implementation for MiniStS game-playing agent.

This module provides the LLMBot class which uses the SATURN service discovery
protocol to connect to AI services on the local network. It supports any
SATURN-compatible service (OpenRouter, Ollama, etc.) with access to 343+ models.

Supported Model Examples:
- openrouter/auto - Intelligent routing to best model (recommended)
- GPT-4 family (gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini)
- Reasoning models (o1-preview, o1-mini, o3-mini)
- Anthropic Claude (claude-3.5-sonnet, claude-3-opus)
- Google Gemini (gemini-pro-1.5, gemini-flash-1.5)
- Meta Llama (llama-3.1-70b-instruct, llama-3.1-405b-instruct)
- And 343+ other models available through OpenRouter

Usage Example:
    bot = LLMBot(LLMBot.ModelName.OPENROUTER_AUTO)

Key Features:
- Zero-configuration service discovery via mDNS
- Automatic failover to backup services
- Provider-agnostic (works with any SATURN service)
- Structured outputs for near-zero invalid response rates
- Reasoning model support with automatic constraint handling
- Multiple prompt strategies (CoT, DAG, etc.)

Migration from Direct API:
- No more API key management (handled by SATURN server)
- No more rate limiting (handled by SATURN server)
- No more provider-specific code
- Access to all OpenRouter models by simply specifying the model name
"""

from __future__ import annotations
import time
import json
from enum import StrEnum
from ggpa.ggpa import GGPA
from action.action import EndAgentTurn, PlayCard
from utility import get_unique_filename
from ggpa.prompt2 import PromptOption, get_action_prompt,\
    get_agent_target_prompt, get_card_target_prompt,\
    strip_response, _get_game_context
from joey_setup.saturn_service_manager import SaturnServiceManager
from typing import TYPE_CHECKING, Any, Optional
from joey_setup.game_schemas import (
    GameAction, OptionSelection, TargetSelection, ActionType,
    get_option_selection_schema, get_target_selection_schema,
    parse_option_selection, parse_target_selection
)
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card
    from action.action import Action

class LLMBot(GGPA):
    class ModelName(StrEnum):
        # OpenRouter intelligent routing (recommended)
        OPENROUTER_AUTO = "openrouter/auto"

        # GPT-4 family (via OpenRouter)
        GPT_4 = "gpt-4"
        GPT_Turbo_4 = "gpt-4-1106-preview"
        GPT_Turbo_35 = "gpt-3.5-turbo"

        # GPT-4o family - optimized for speed and cost
        GPT_4o = "gpt-4o"
        GPT_4o_mini = "gpt-4o-mini"

        # Reasoning models - excel at strategic planning and complex decisions
        o1_preview = "o1-preview"
        o1_mini = "o1-mini"
        o3_mini = "o3-mini"

        # Additional OpenRouter model examples (343+ models available)
        # Anthropic Claude models
        CLAUDE_35_SONNET = "anthropic/claude-3.5-sonnet"
        CLAUDE_3_OPUS = "anthropic/claude-3-opus"

        # Google Gemini models
        GEMINI_PRO_15 = "google/gemini-pro-1.5"
        GEMINI_FLASH_15 = "google/gemini-flash-1.5"

        # Meta Llama models
        LLAMA_31_70B = "meta-llama/llama-3.1-70b-instruct"
        LLAMA_31_405B = "meta-llama/llama-3.1-405b-instruct"

    # Model categorization for API handling
    CHAT_MODELS = [
        ModelName.GPT_4, ModelName.GPT_Turbo_4, ModelName.GPT_Turbo_35,
        ModelName.GPT_4o, ModelName.GPT_4o_mini
    ]
    REASONING_MODELS = [ModelName.o1_preview, ModelName.o1_mini, ModelName.o3_mini]

    # Models that support structured outputs
    STRUCTURED_OUTPUT_MODELS = [
        ModelName.GPT_4o, ModelName.GPT_4o_mini,
        ModelName.GPT_Turbo_4
    ]

    def is_reasoning_model(self) -> bool:
        return self.model_name in LLMBot.REASONING_MODELS

    def supports_structured_output(self) -> bool:
        return self.model_name in LLMBot.STRUCTURED_OUTPUT_MODELS

    def convert_system_to_user(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not messages:
            return messages

        converted = []
        system_content = ""

        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
            else:
                converted.append(msg.copy())

        # Prepend system content to first user message
        if system_content and converted:
            for msg in converted:
                if msg["role"] == "user":
                    msg["content"] = f"[Instructions]\n{system_content}\n[User Query]\n{msg['content']}"
                    break

        return converted

    def ask_llm(self) -> str:
        """Make an API call via SATURN service and return the response text."""
        try:
            before_request = time.time()

            if self.model_name in LLMBot.REASONING_MODELS:
        
                messages = self.convert_system_to_user(self.messages)
                temperature = None  # Reasoning models don't use temperature
                timeout = 120  
            else:
                # Standard chat models
                messages = self.messages
                temperature = 0.7
                timeout = 30

            # Make request through SATURN service manager (modern implementation)
            response = self.saturn_manager.chat_completion(
                messages=messages,
                model=str(self.model_name),
                max_tokens=500,
                temperature=temperature,
                timeout=timeout
            )

            if response is None:
                raise Exception("SATURN service request failed - no response from any service")

            elapsed = time.time() - before_request
            self.metadata["response_time"].append(elapsed)
            print(f"[LLM] Response time: {elapsed:.2f}s")

            # Track token usage if available
            if 'usage' in response:
                tokens_used = response['usage'].get('total_tokens', 0)
                if 'total_tokens' not in self.metadata:
                    self.metadata['total_tokens'] = 0
                self.metadata['total_tokens'] += tokens_used

            # Record request and response in history
            self.history.append({
                'messages': messages,
                'response': response,
                'response_time': elapsed,
                'tokens_used': response.get('usage', {}).get('total_tokens', 0)
            })

            return response['choices'][0]['message']['content']

        except Exception as e:
            print(f"[LLM] Error making SATURN request: {e}")
            raise e


    def translate_to_string_input(self, openai_messages: list[dict[str, str]]):

        return "\n".join([message["content"] for message in openai_messages])

    def __init__(
        self,
        model_name: LLMBot.ModelName,
        prompt_option: PromptOption,
        few_shot: int,
        show_option_results: bool,
        use_structured_output: bool = False,
        saturn_manager: Optional[SaturnServiceManager] = None
    ):
        """
        Initialize the LLM bot with SATURN service discovery.

        Args:
            model_name: Which model to use (e.g., gpt-4o, gpt-4-turbo)
            prompt_option: Prompting strategy (CoT, DAG, etc.)
            few_shot: Number of previous examples to include in context
            show_option_results: Whether to show option outcomes in prompts
            use_structured_output: Whether to use structured JSON output format
            saturn_manager: SATURN service manager (creates one if not provided)
        """
        model_name_dict = {
            LLMBot.ModelName.GPT_4: '4',
            LLMBot.ModelName.GPT_Turbo_4: 't4',
            LLMBot.ModelName.GPT_Turbo_35: 't35',
            LLMBot.ModelName.GPT_4o: '4o',
            LLMBot.ModelName.GPT_4o_mini: '4o-mini',
            LLMBot.ModelName.o1_preview: 'o1-pre',
            LLMBot.ModelName.o1_mini: 'o1-mini',
            LLMBot.ModelName.o3_mini: 'o3-mini',
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
        self.use_structured_output = use_structured_output
        self.messages: list[dict[str, str]] = []

        # Initialize SATURN service manager for zero-config AI service discovery
        if saturn_manager is None:
            print(f"[LLM] Initializing SATURN service discovery...")
            self.saturn_manager = SaturnServiceManager(discovery_timeout=3.0)
            self._owns_saturn_manager = True
        else:
            self.saturn_manager = saturn_manager
            self._owns_saturn_manager = False

        # Build name with optional structured output indicator
        name_suffix = '-struct' if use_structured_output else ''
        super().__init__(f"LLM-{model_name_dict[model_name]}-{prompt_dict[prompt_option]}-f{self.few_shot}{'-results' if show_option_results else ''}{name_suffix}")
        self.clear_metadata()
        self.clear_history()

    def parse_structured_response(self, response: str, min_val: int, max_val: int) -> Optional[int]:
        try:
            parsed = parse_option_selection(response)
            value = parsed.selected_index
            if value >= min_val and value <= max_val:
                return value
            else:
                print(f'Wrong range for {self.name}: Value {value} not in [{min_val}, {max_val}]')
                self.metadata["wrong_range_count"] += 1
                return None
        except Exception as e:
            print(f"Failed to parse structured response: {e}\nResponse: {response}")
            self.metadata["wrong_format_count"] += 1
            return None

    def get_integer_response(self, min: int, max: int, prompt_option: PromptOption) -> int:
        if max == min:
            self.history.append({'auto-answer': str(min)})
            self.messages = self.messages[:-1]
            return min

        while True:
            try:
                response: str = self.ask_llm()
            except Exception as e:
                print(e)
                continue

            # Try structured parsing first if enabled
            if self.use_structured_output and self.supports_structured_output():
                value = self.parse_structured_response(response, min, max)
                if value is not None:
                    break
                continue

            # Fall back to text parsing
            try:
                value = int(strip_response(response, prompt_option))
                if value >= min and value <= max:
                    break
            except Exception as e:
                print(f"Wrong format for {self.name}, option {prompt_option}:\n*{response}*\nExtracted: *{strip_response(response, prompt_option)}*")
                self.metadata["wrong_format_count"] += 1
                continue
            print(f'Wrong range for {self.name}: *{response}*\nValue: {value}')
            self.metadata["wrong_range_count"] += 1

        self.messages.append({"role": "assistant", "content": response})
        return value

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> EndAgentTurn|PlayCard:
        """
        Choose which card to play or whether to end the turn.
        Args:
            game_state: Current game state
            battle_state: Current battle state
        Returns:
            PlayCard action or EndAgentTurn action
        """
        options = self.get_choose_card_options(game_state, battle_state)
        get_context = False

        # Build initial messages
        if self.few_shot == 0:
            if self.is_reasoning_model():
                # Reasoning models: simpler instructions without system message
                self.messages: list[dict[str, str]] = [
                    {"role": "user", "content": "You are a bot specialized in playing a card game. Respond with the number of your chosen action."}
                ]
            else:
                self.messages: list[dict[str, str]] = [
                    {"role": "system", "content": "You are a bot specialized in playing a card game."}
                ]
            get_context = True
        elif len(self.messages) == 0:
            if self.is_reasoning_model():
                context = _get_game_context(game_state, battle_state, options)
                self.messages: list[dict[str, str]] = [
                    {"role": "user", "content": f"You are a bot specialized in playing a card game.\n\n{context}"}
                ]
            else:
                self.messages: list[dict[str, str]] = [
                    {"role": "system", "content": "You are a bot specialized in playing a card game."},
                    {"role": "user", "content": _get_game_context(game_state, battle_state, options)}
                ]

        # Manage conversation history for few-shot
        if len(self.messages) - 2 + 2 > self.few_shot * 2:
            self.messages = self.messages[:2] + self.messages[-(self.few_shot-1)*2:]

        prompt = get_action_prompt(game_state, battle_state, options, self.prompt_option, get_context, self.show_option_results)

        # Add structured output instructions if enabled
        if self.use_structured_output and self.supports_structured_output():
            prompt += "\n\nRespond with JSON: {\"selected_index\": <number>, \"reasoning\": \"<brief explanation>\"}"

        self.messages.append({"role": "user", "content": prompt})
        value = self.get_integer_response(0, len(options)-1, self.prompt_option)
        return options[value]

    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        prompt = get_agent_target_prompt(battle_state, list_name, agent_list)

        if self.use_structured_output and self.supports_structured_output():
            prompt += "\n\nRespond with JSON: {\"selected_index\": <number>, \"reasoning\": \"<brief explanation>\"}"

        self.messages.append({"role": "user", "content": prompt})
        value = self.get_integer_response(0, len(agent_list)-1, PromptOption.NONE)
        return agent_list[value]

    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        prompt = get_card_target_prompt(battle_state, list_name, card_list)

        if self.use_structured_output and self.supports_structured_output():
            prompt += "\n\nRespond with JSON: {\"selected_index\": <number>, \"reasoning\": \"<brief explanation>\"}"

        self.messages.append({"role": "user", "content": prompt})
        value = self.get_integer_response(0, len(card_list)-1, PromptOption.NONE)
        return card_list[value]

    def dump_history(self, filename: str):
        """Save conversation history to a JSON file."""
        filename = get_unique_filename(filename, 'json')
        with open(filename, "w") as file:
            json.dump(self.history, file, indent=4)

    def dump_metadata(self, filename: str):
        """Append metadata to a file."""
        print(filename)
        with open(filename, "a") as file:
            json.dump(self.metadata, file, indent=4)
            file.write('\n')

    def clear_metadata(self):
        """Reset metadata counters."""
        self.metadata["response_time"] = []
        self.metadata["wrong_format_count"] = 0
        self.metadata["wrong_range_count"] = 0
        self.metadata["total_tokens"] = 0

    def clear_history(self):
        """Reset conversation history."""
        self.history: list[dict[str, Any]] = [{
            'model': str(self.model_name),
            'prompt_option': str(self.prompt_option),
            'use_structured_output': self.use_structured_output,
        }]

    def cleanup(self):
        """Clean up SATURN service manager resources."""
        if self._owns_saturn_manager:
            print(f"[LLM] Closing SATURN service manager")
            self.saturn_manager.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
