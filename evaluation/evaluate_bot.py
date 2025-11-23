from tqdm import tqdm
import pandas as pd
import time
import argparse
from typing import Callable
from joblib import delayed, Parallel
import sys
import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import AcidSlimeSmall, SpikeSlimeSmall, JawWorm, Goblin, HobGoblin, Leech, Enemy
from card import CardGen, Card, CardRepo
from ggpa.ggpa import GGPA
from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.chatgpt_bot import ChatGPTBot
from ggpa.prompt2 import PromptOption
# --- NEW IMPORTS: None and Basic LLM Agents ---
from ggpa.none_agent import NoneAgent
from ggpa.basic_agent import BasicAgent
# ----------------------------------------------

def name_to_bot(name: str, limit_share: float) -> GGPA:
    # New agent types: NoneAgent and BasicAgent
    if name == 'none':
        return NoneAgent()
    if name == 'basic':
        return BasicAgent()
    
    if name == 'r':
        return RandomBot()
    if len(name) > 3 and name[0:3] == 'bts':
        depth = int(name[3:])
        return BacktrackBot(depth, True)
    if len(name) > 2 and name[0:2] == 'bt':
        depth = int(name[2:])
        return BacktrackBot(depth, False)
    if len(name) > 3 and name[:3] == 'gpt':
        show_results = False
        if '-results' in name:
            name = name[:-len('-results')]
            show_results = True
        if len(name.split('-')) == 3:
            name += '-f0'
        _, model, prompt, fs = name.split('-')
        model_dict: dict[str, ChatGPTBot.ModelName] = {
            't3.5': ChatGPTBot.ModelName.GPT_Turbo_35,
            '4': ChatGPTBot.ModelName.GPT_4,
            't4': ChatGPTBot.ModelName.GPT_Turbo_4,
            'it3.5': ChatGPTBot.ModelName.Instruct_GPT_Turbo_35,
            'idav': ChatGPTBot.ModelName.Instruct_Davinci,
        }
        prompt_dict: dict[str, PromptOption] = {
            'none': PromptOption.NONE,
            'dag': PromptOption.DAG,
            'cot': PromptOption.CoT,
            'cotr': PromptOption.CoT_rev,
        }
        fs = int(fs[1:])
        return ChatGPTBot(model_dict[model], prompt_dict[prompt], fs, show_results, limit_share)
    raise Exception(f"Bot name '{name}' not recognized")

# ... (Rest of the file remains unchanged)