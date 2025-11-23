"""
Pydantic schemas for structured outputs in MiniStS game actions.

This module defines the data models used for OpenAI's Structured Output feature,
which ensures LLM responses match expected formats exactly, reducing parsing errors
from ~80% to near-zero.

Key Features:
- GameAction: Primary schema for card game move selection
- TargetSelection: Schema for targeting enemies or cards
- Support for both Chat Completions API and Responses API

Future Extensions:
- Add schemas for multi-provider support (Claude, Gemini, LLAMA)
- Add schemas for more complex game actions (potions, relics, shop)
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ActionType(str, Enum):
    PLAY_CARD = "play_card"
    END_TURN = "end_turn"


class GameAction(BaseModel):
    """
    Schema for a single game action response from the LLM.

    Attributes:
        action_type: Whether to play a card or end the turn
        card_index: Index of card to play (0-based), None if ending turn
        target_index: Target enemy index if card requires target, None otherwise
        reasoning: Brief explanation of the action choice (optional)

    Example JSON response:
        {
            "action_type": "play_card",
            "card_index": 2,
            "target_index": 0,
            "reasoning": "Playing Strike to deal damage to the enemy"
        }
    """
    action_type: ActionType = Field(
        ...,
        description="Type of action: 'play_card' to play a card, 'end_turn' to end your turn"
    )
    card_index: Optional[int] = Field(
        None,
        description="Index of card to play from hand (0-based). Required if action_type is 'play_card', otherwise should be null.",
        ge=0
    )
    target_index: Optional[int] = Field(
        None,
        description="Index of target enemy (0-based) if the card requires a target. Null for cards that don't require targeting or for end_turn.",
        ge=0
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why this action was chosen. Optional but helpful for debugging."
    )


class OptionSelection(BaseModel):

    selected_index: int = Field(
        ...,
        description="Index of the selected option (0-based)",
        ge=0
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why this option was selected"
    )


class TargetSelection(BaseModel):
    target_index: int = Field(
        ...,
        description="Index of the selected target (0-based)",
        ge=0
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of the target selection"
    )


class MultiTargetAction(BaseModel):
    """
    Example JSON response:
        {
            "action_type": "play_card",
            "card_index": 1,
            "target_indices": [0, 2],
            "reasoning": "Using Cleave to hit multiple enemies"
        }
    """
    action_type: ActionType = Field(
        ...,
        description="Type of action being taken"
    )
    card_index: Optional[int] = Field(
        None,
        description="Index of card to play from hand (0-based)",
        ge=0
    )
    target_indices: list[int] = Field(
        default_factory=list,
        description="List of target indices for multi-target effects"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of the action choice"
    )


# Schema definitions for OpenAI's response_format parameter
# These are used directly in API calls

def get_game_action_schema() -> dict:
    """

    Example usage:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": get_game_action_schema()
            }
        )
    """
    return {
        "name": "game_action",
        "strict": True,
        "schema": GameAction.model_json_schema()
    }


def get_option_selection_schema() -> dict:
    """
    Get the JSON schema for OptionSelection, formatted for OpenAI's response_format.

    Returns:
        Dictionary containing the schema configuration for structured outputs.
    """
    return {
        "name": "option_selection",
        "strict": True,
        "schema": OptionSelection.model_json_schema()
    }


def get_target_selection_schema() -> dict:

    return {
        "name": "target_selection",
        "strict": True,
        "schema": TargetSelection.model_json_schema()
    }


# Utility functions for parsing structured responses

def parse_game_action(response_content: str) -> GameAction:

    import json
    data = json.loads(response_content)
    return GameAction.model_validate(data)


def parse_option_selection(response_content: str) -> OptionSelection:

    import json
    data = json.loads(response_content)
    return OptionSelection.model_validate(data)


def parse_target_selection(response_content: str) -> TargetSelection:
  
    import json
    data = json.loads(response_content)
    return TargetSelection.model_validate(data)


# Type hints for use in other modules
GameActionSchema = GameAction
OptionSelectionSchema = OptionSelection
TargetSelectionSchema = TargetSelection
