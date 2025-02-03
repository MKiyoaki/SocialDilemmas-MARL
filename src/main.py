import torch
import os
import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3

# Initialize environment
env = simple_spread_v3.env(render_mode="human")
# Correct way to handle env.reset() (it returns a tuple in this case)
env.reset(seed=42)


# Model path
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/models")
model_dir = os.path.join(model_dir, "mappo_seed366422125_pz-mpe-simple-spread-v3_2024-12-01_17-17/10000250/")
model_dir = os.path.join(model_dir, "agent.th")



# Load the trained model
model = torch.load(model_dir)

# Reset environment and get initial observations
env = simple_spread_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

# Close the environment
env.close()
