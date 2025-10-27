import os
from config import Character, Verbose
from game import GameState
from battle import BattleState
from agent import JawWorm
from rcot_agent import RCotAgent, RCotConfig
import dotenv

dotenv.load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY environment variable"

config = RCotConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=500,
    retry_limit=3
)

agent = RCotAgent(config)

game_state = GameState(Character.IRON_CLAD, agent, ascention=0)
battle_state = BattleState(
    game_state,
    JawWorm(game_state),
    verbose=Verbose.LOG
)

print("Starting battle with RCoT agent using Responses API...")
battle_state.run()

if battle_state.get_end_result() == 1:
    print("\n=== VICTORY ===")
else:
    print("\n=== DEFEAT ===")

print("\nAgent Statistics:")
for key, value in agent.get_statistics().items():
    print(f"  {key}: {value}")

agent.dump_history("rcot_game_history.json")