import torch
import mjx
from mjx.agents import RandomAgent
from mortal_like_agent import MortalLikeAgent

CKPT_PATH = "mortal_like_agent_weight.pt"

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

print("mjx import: OK")
print(f"device: {device}")

my_agent = MortalLikeAgent(CKPT_PATH, device=device)
random_agent = RandomAgent()

print(f"MortalLikeAgent loaded: {CKPT_PATH}")

agents = {
    "player_0": my_agent,
    "player_1": random_agent,
    "player_2": random_agent,
    "player_3": random_agent,
}

env = mjx.MjxEnv()
obs_dict = env.reset()

step = 0
while not env.done():
    actions = {pid: agents[pid].act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)
    step += 1

returns = env.rewards()
print(f"Game finished in {step} steps")
print(f"Returns: {returns}")