import sys

import torch
import os
import gymnasium as gym
from shimmy import MeltingPotCompatibilityV0


def render_env(substrate_name, timesteps=int(1e5)):
    # Initialize environment
    env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode="human")

    for _ in range(timesteps):
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

    # Close the environment
    env.close()

if __name__ == "__main__":
    render_env(sys.argv[1])
    # render_env("collaborative_cooking__circuit")
    # render_env("collaborative_cooking__asymmetric")


