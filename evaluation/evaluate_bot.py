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
from ggpa.none_agent import NoneAgent
from ggpa.basic_agent import BasicAgent
from g3_files.agents.llm_bot import SimpleLLMBot
from g3_files.agents.mcts_bot import MCTSAgent
from g3_files.agents.rcot_agent import RCotAgent, RCotConfig

def name_to_bot(name: str, limit_share: float) -> GGPA:
    # Baseline agents: NoneAgent and BasicAgent
    if name == 'none':
        return NoneAgent()
    if name == 'basic':
        return BasicAgent()

    # G3 MCTS agents with configurable iterations
    if name == 'g3-mcts':
        return MCTSAgent(iterations=100)
    if name.startswith('g3-mcts-'):
        iterations = int(name.split('-')[-1])
        return MCTSAgent(iterations=iterations)

    # G3 RCoT agents (Reflective Chain-of-Thought)
    if name == 'g3-rcot-cot':
        return RCotAgent(RCotConfig(prompt_option="cot"))
    if name == 'g3-rcot-none':
        return RCotAgent(RCotConfig(prompt_option="none"))
    if name == 'g3-rcot-rcot':
        return RCotAgent(RCotConfig(prompt_option="rcot"))

    # G3 SimpleLLMBot agents with direct OpenRouter API calls
    if name == 'g3-llm-openrouter-auto':
        return SimpleLLMBot(SimpleLLMBot.ModelName.OPENROUTER_AUTO, PromptOption.CoT, 0, False)
    if name == 'g3-llm-gpt4o':
        return SimpleLLMBot(SimpleLLMBot.ModelName.GPT_4o, PromptOption.CoT, 0, False)
    if name == 'g3-llm-claude':
        return SimpleLLMBot(SimpleLLMBot.ModelName.CLAUDE_35_SONNET, PromptOption.CoT, 0, False)
    if name == 'g3-llm-gemini':
        return SimpleLLMBot(SimpleLLMBot.ModelName.GEMINI_PRO_15, PromptOption.CoT, 0, False)

    # Free models
    if name == 'g3-llm-llama-free':
        return SimpleLLMBot(SimpleLLMBot.ModelName.LLAMA_33_70B_FREE, PromptOption.CoT, 0, False)
    if name == 'g3-llm-qwen-free':
        return SimpleLLMBot(SimpleLLMBot.ModelName.QWEN3_235B_FREE, PromptOption.CoT, 0, False)
    if name == 'g3-llm-nemotron-free':
        return SimpleLLMBot(SimpleLLMBot.ModelName.NEMOTRON_NANO_FREE, PromptOption.CoT, 0, False)
    if name == 'g3-llm-gpt-oss-free':
        return SimpleLLMBot(SimpleLLMBot.ModelName.GPT_OSS_20B_FREE, PromptOption.CoT, 0, False)
    if name == 'g3-llm-deepseek-free':
        return SimpleLLMBot(SimpleLLMBot.ModelName.DEEPSEEK_R1T2_FREE, PromptOption.CoT, 0, False)

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
    raise Exception("Bot name not recognized")

def get_scenario(index: int, anonymize: bool) -> Callable[[], tuple[str, list[Card]]]:
    scenario = []
    if index == 0:
        scenario = CardRepo.get_scenario_0
    elif index == 1:
        scenario = CardRepo.get_scenario_1
    elif index == 2:
        scenario = CardRepo.get_scenario_2
    elif index == 3:
        scenario = CardRepo.get_scenario_3
    elif index == 4:
        scenario = CardRepo.get_scenario_4
    else:
        raise Exception(f"Scenario not recognized: {index}")
    if anonymize:
        return lambda: CardRepo.anonymize_scenario(scenario())
    return scenario

def get_enemies(enemies: str, game_state: GameState) -> list[Enemy]:
    ret: list[Enemy] = []
    for char in enemies:
        if char == 'g':
            ret.append(Goblin(game_state))
        elif char == 'h':
            ret.append(HobGoblin(game_state))
        elif char == 'l':
            ret.append(Leech(game_state))
        elif char == 'j':
            ret.append(JawWorm(game_state))
        else:
            raise Exception(f"Enemies not recognized for {char} in {enemies}")
    return ret

def simulate_one(index: int, bot: GGPA, deck: list[Card], enemies: str, path: str, verbose: Verbose):
    try:
        game_state = GameState(Character.IRON_CLAD, bot, 0)
        game_state.set_deck(*deck)
        battle_state = BattleState(game_state, *get_enemies(enemies, game_state),
                                   verbose=verbose, log_filename=os.path.join(path, f'{index}_{bot.name}'))
        battle_state.run()
        if isinstance(bot, ChatGPTBot):
            bot.dump_history(os.path.join(path, f'{index}_{bot.name}_history'))
            bot.dump_metadata(os.path.join(path, f'{bot.name}_metadata'))
        return [bot.name, game_state.player.health, game_state.get_end_results() != -1]
    except Exception as e:
        # Convert exceptions to picklable format for joblib
        # API errors from OpenAI, Anthropic, etc. can't be pickled properly
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error in simulation {index} for {bot.name}: {error_msg}")
        # Return a loss result with 0 health when an error occurs
        return [bot.name, 0, False]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_count', type=int)
    parser.add_argument('thread_count', type=int)
    parser.add_argument('scenario', type=int)
    parser.add_argument('enemies', type=str)
    parser.add_argument('bots', nargs='+')
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--dir', type=str, default="")
    parser.add_argument('--log', action=argparse.BooleanOptionalAction)
    parser.add_argument('--anonymize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--time', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    test_count = args.test_count
    thread_count = args.thread_count
    anonymize = args.anonymize
    time_execution = args.time
    scenario = get_scenario(args.scenario, anonymize)
    scenario_name, _ = scenario()
    if anonymize:
        scenario_name += "-anon"
    enemies = args.enemies
    custom_name = args.name
    custom_dir = args.dir
    verbose = Verbose.LOG if args.log else Verbose.NO_LOG
    bots: list[GGPA] = [name_to_bot(name, 1/thread_count) for name in args.bots]  
    bot_names = '_'.join([bot.name for bot in bots])
    dir_name = f'{custom_name}_{scenario_name}_enemies_{enemies}_{test_count}_boteval'
    if custom_dir != "":
        dir_name = os.path.join(custom_dir, dir_name)
    path = os.path.join('evaluation_results', dir_name)
    os.makedirs(path)
    print(f'simulating {test_count} times, for {bot_names} - {thread_count} threads')
    print(f'results can be found at {path}')
    if not time_execution:
        results_dataset = Parallel(n_jobs=thread_count)(delayed(simulate_one)(i, bots[i//test_count], scenario()[1], enemies, path, verbose) for i in tqdm(range(test_count * len(bots))))
    else:
        results_dataset = []
        execution_times = {}
        for bot_id in range(len(bots)):
            start_time = time.time()
            results_dataset += Parallel(n_jobs=thread_count)(delayed(simulate_one)(i, bots[bot_id], scenario()[1], enemies, path, verbose) for i in tqdm(range(test_count * bot_id, test_count * (bot_id + 1))))
            execution_times[bots[bot_id].name] = {'avg_execution': (time.time() - start_time)/test_count}
            import json
            with open(os.path.join(path, "execution_times_partial.json"), "a") as fp:
                json.dump(execution_times , fp) 
    assert isinstance(results_dataset, list), "Parallel jobs have not resulted in an output of type list"
    df = pd.DataFrame(
        results_dataset,
        columns=["BotName", "PlayerHealth", "Win"]
    )
    df.to_csv(os.path.join(path, f"results.csv"), index=False)
    if time_execution:
        import json
        with open(os.path.join(path, "execution_times.json"), "w") as fp:
            json.dump(execution_times , fp) 

if __name__ == '__main__':
    main()