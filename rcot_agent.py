from __future__ import annotations
import os
import time
import json
import re
from typing import TYPE_CHECKING, Any, Optional
import random
from dataclasses import dataclass, field, asdict

from openai import OpenAI
from ggpa.ggpa import GGPA
from ggpa.prompt2 import (
    PromptOption, 
    get_agent_target_prompt, 
    get_card_target_prompt,
    _get_game_context
)
from action.action import EndAgentTurn, PlayCard
from utility import get_unique_filename

if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card


@dataclass
class RCotConfig:
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 500
    anonymize_cards: bool = False
    retry_limit: int = 3
    

@dataclass
class RCotStatistics:
    total_requests: int = 0
    invalid_responses: int = 0
    wrong_format_count: int = 0
    wrong_range_count: int = 0
    total_tokens_used: int = 0
    response_times: list[float] = field(default_factory=list)
    
    @property
    def invalid_rate_percent(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round((self.invalid_responses / self.total_requests) * 100, 2)
    
    @property
    def avg_tokens_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_tokens_used / self.total_requests, 2)
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return round(sum(self.response_times) / len(self.response_times), 3)


class RCotAgent(GGPA):
    
    def __init__(self, config: Optional[RCotConfig] = None):
        self.config = config or RCotConfig()
        super().__init__(f"RCoT-{self.config.model.replace('gpt-', '')}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        
        self.stats = RCotStatistics()
        self.history: list[dict[str, Any]] = []
        self.previous_response_id: Optional[str] = None
        
        self.system_prompt = (
            "You are an expert card game strategist playing a deck-building roguelike. "
            "When making decisions, consider:\n"
            "1. Card synergies and deck cycling\n"
            "2. Enemy attack patterns and intent\n"
            "3. Resource management (HP, block, mana)\n"
            "4. Status effects and their durations\n"
            "5. Short-term vs long-term value\n\n"
            "CRITICAL: Your response MUST start with ONLY a number on the first line.\n"
            "Format:\n"
            "Line 1: <move_index>\n"
            "Following lines: Your strategic reasoning\n\n"
            "Example:\n"
            "3\n"
            "Playing the block card because the enemy intends to attack for 12 damage next turn..."
        )
    
    def _should_include_context(self, battle_state: BattleState) -> bool:
        return (
            self.previous_response_id is None or
            battle_state.turn == 1 or 
            battle_state.turn % 5 == 1
        )
    
    def _analyze_deck_state(self, battle_state: BattleState) -> str:
        lines = ["=== DECK STATE ==="]
        
        total_cards = len(battle_state.draw_pile) + len(battle_state.discard_pile) + len(battle_state.hand)
        lines.append(f"Total deck: {total_cards} cards")
        lines.append(f"Draw pile: {len(battle_state.draw_pile)} cards")
        lines.append(f"Discard: {len(battle_state.discard_pile)} cards")
        lines.append(f"Hand: {len(battle_state.hand)} cards")
        lines.append(f"Exhausted: {len(battle_state.exhaust_pile)} cards")
        
        if battle_state.draw_pile or battle_state.discard_pile:
            lines.append("\nRemaining cards in cycle:")
            remaining = battle_state.draw_pile + battle_state.discard_pile
            card_names = {}
            for card in remaining:
                name = card.name if hasattr(card, 'name') else str(card)
                card_names[name] = card_names.get(name, 0) + 1
            
            for name, count in sorted(card_names.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {name}: {count}x")
        
        return "\n".join(lines)
    
    def _analyze_enemy_patterns(self, game_state: GameState, battle_state: BattleState) -> str:
        lines = ["=== ENEMY ANALYSIS ==="]
        
        for i, enemy in enumerate(battle_state.enemies):
            lines.append(f"\nEnemy {i} ({enemy.name}):")
            lines.append(f"  Current intent: {enemy.get_intention(game_state, battle_state)}")
            
            if hasattr(enemy, 'action_set'):
                action_set = enemy.action_set
                lines.append(f"  Action pattern: {type(action_set).__name__}")
        
        return "\n".join(lines)
    
    def _build_detailed_card_info(self, card, index: int, battle_state: BattleState) -> str:
        card_name = f"Card_{index}" if self.config.anonymize_cards else card.name
        
        info_parts = [f"{index}. {card_name} (Cost: {card.mana_cost.peek()})"]
        
        if hasattr(card, 'get_description'):
            info_parts.append(f"   Effect: {card.get_description()}")
        elif hasattr(card, 'description'):
            info_parts.append(f"   Effect: {card.description}")
        elif hasattr(card, '__str__'):
            effect = str(card)
            if effect and effect != card_name:
                info_parts.append(f"   {effect}")
        
        player = battle_state.player
        if hasattr(player.status_effect_state, '__str__'):
            status_str = str(player.status_effect_state)
            if 'weak' in status_str.lower() and 'attack' in card_name.lower():
                info_parts.append(f"   NOTE: Weakened (attack damage reduced)")
            if 'strength' in status_str.lower() and 'attack' in card_name.lower():
                info_parts.append(f"   NOTE: Strengthened (attack damage increased)")
        
        return "\n".join(info_parts)
    
    def _build_prompt(self, game_state: GameState, battle_state: BattleState, 
                     options: list[PlayCard | EndAgentTurn]) -> str:
        prompt_parts = []
        
        prompt_parts.append(self._build_current_state(game_state, battle_state))
        prompt_parts.append(self._analyze_deck_state(battle_state))
        prompt_parts.append(self._analyze_enemy_patterns(game_state, battle_state))
        
        prompt_parts.append("\n=== YOUR OPTIONS ===")
        for i, option in enumerate(options):
            if isinstance(option, PlayCard):
                card = battle_state.hand[option.card_index]
                prompt_parts.append(self._build_detailed_card_info(card, i, battle_state))
            else:
                prompt_parts.append(f"{i}. End Turn")
        
        prompt_parts.append("\n=== DECISION REQUEST ===")
        prompt_parts.append(
            f"Choose the best option (0-{len(options)-1}).\n"
            "RESPOND WITH THE NUMBER FIRST, THEN EXPLAIN YOUR REASONING."
        )
        
        return "\n".join(prompt_parts)
    
    def _build_current_state(self, game_state: GameState, battle_state: BattleState) -> str:
        player = battle_state.player
        lines = [
            f"=== TURN {battle_state.turn} ===",
            f"Mana: {battle_state.mana}/{game_state.max_mana}",
            f"Player: HP {player.health}/{player.max_health}, Block {player.block}",
            f"Status: {repr(player.status_effect_state)}",
            "\nEnemies:"
        ]
        
        for i, enemy in enumerate(battle_state.enemies):
            lines.append(
                f"  {i}. {enemy.name}: HP {enemy.health}/{enemy.max_health}, "
                f"Block {enemy.block}, Intent: {enemy.get_intention(game_state, battle_state)}"
            )
        
        return "\n".join(lines)
    
    def _parse_response(self, content: str, max_index: int) -> Optional[int]:
        content = ''.join(c if c not in '.-' else ' ' for c in content)
        lines = content.split()
        lines = [line for line in lines if len(line) > 0]
        
        if not lines:
            return None
        
        try:
            value = int(lines[0])
            if 0 <= value < max_index:
                return value
            self.stats.wrong_range_count += 1
            return None
        except ValueError:
            self.stats.wrong_format_count += 1
            return None
    
    def _make_api_call(self, user_message: str) -> Optional[dict]:
        try:
            start_time = time.time()
            
            input_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            kwargs = {
                "model": self.config.model,
                "input": input_messages,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens
            }
            
            if self.previous_response_id:
                kwargs["previous_response_id"] = self.previous_response_id
            
            response = self.client.responses.create(**kwargs)
            
            elapsed = time.time() - start_time
            
            self.stats.total_requests += 1
            self.stats.response_times.append(elapsed)
            
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.stats.total_tokens_used += tokens_used
            
            return response
            
        except Exception as e:
            print(f"API call failed: {e}")
            self.stats.invalid_responses += 1
            return None
    
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        options = self.get_choose_card_options(game_state, battle_state)
        
        user_prompt = self._build_prompt(game_state, battle_state, options)
        
        if self._should_include_context(battle_state):
            context = _get_game_context(game_state, battle_state, options)
            user_prompt = f"{context}\n\n{user_prompt}"
        
        for attempt in range(self.config.retry_limit):
            response = self._make_api_call(user_prompt)
            
            if response is None:
                print(f"Attempt {attempt + 1}/{self.config.retry_limit} failed")
                time.sleep(1)
                continue
            
            content = response.output_text.strip()
            move_index = self._parse_response(content, len(options))
            
            if move_index is not None:
                self.previous_response_id = response.id
                
                self.history.append({
                    'turn': battle_state.turn,
                    'prompt': user_prompt,
                    'response': content,
                    'move_index': move_index,
                    'response_id': response.id,
                    'tokens': response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
                })
                
                return options[move_index]
            
            print(f"Invalid response format on attempt {attempt + 1}")
            self.stats.invalid_responses += 1
        
        print("All attempts failed, using fallback logic")
        return self._fallback_choice(options, battle_state)
    
    def _fallback_choice(self, options: list[PlayCard | EndAgentTurn], 
                        battle_state: BattleState) -> PlayCard | EndAgentTurn:
        playable = [opt for opt in options if isinstance(opt, PlayCard)]
        
        if playable:
            return random.choice(playable)
        
        return options[-1]
    
    def choose_agent_target(self, battle_state: BattleState, list_name: str, 
                          agent_list: list[Agent]) -> Agent:
        if len(agent_list) == 1:
            return agent_list[0]
        
        prompt = get_agent_target_prompt(battle_state, list_name, agent_list)
        
        response = self._make_api_call(prompt)
        if response:
            content = response.output_text.strip()
            try:
                index = int(re.findall(r'\d+', content)[0])
                if 0 <= index < len(agent_list):
                    self.previous_response_id = response.id
                    return agent_list[index]
            except (ValueError, IndexError):
                pass
        
        return min(agent_list, key=lambda a: a.health)
    
    def choose_card_target(self, battle_state: BattleState, list_name: str, 
                          card_list: list[Card]) -> Card:
        if len(card_list) == 1:
            return card_list[0]
        
        prompt = get_card_target_prompt(battle_state, list_name, card_list)
        
        response = self._make_api_call(prompt)
        if response:
            content = response.output_text.strip()
            try:
                index = int(re.findall(r'\d+', content)[0])
                if 0 <= index < len(card_list):
                    self.previous_response_id = response.id
                    return card_list[index]
            except (ValueError, IndexError):
                pass
        
        return card_list[0]
    
    def get_statistics(self) -> dict[str, Any]:
        stats_dict = asdict(self.stats)
        stats_dict['invalid_rate_percent'] = self.stats.invalid_rate_percent
        stats_dict['avg_tokens_per_request'] = self.stats.avg_tokens_per_request
        stats_dict['avg_response_time'] = self.stats.avg_response_time
        return stats_dict
    
    def dump_history(self, filename: str) -> None:
        filename = get_unique_filename(filename, 'json')
        with open(filename, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'statistics': self.get_statistics(),
                'history': self.history
            }, f, indent=2)
        print(f"RCoT: History saved to {filename}")
    
    def dump_metadata(self, filename: str) -> None:
        metadata = {
            'agent': self.name,
            'config': asdict(self.config),
            'statistics': self.get_statistics(),
            'timestamp': time.time()
        }
        with open(filename, 'a') as f:
            json.dump(metadata, f, indent=2)
            f.write('\n')
        print(f"RCoT: Metadata appended to {filename}")