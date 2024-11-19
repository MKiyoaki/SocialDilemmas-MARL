import functools
import pygame
import numpy as np

from gymnasium import utils as gym_utils
import matplotlib.pyplot as plt

from envs import MultiAgentEnv
from meltingpot.meltingpot import substrate
from ml_collections import config_dict
from pettingzoo.utils import wrappers
from src.utils import utils

PLAYER_STR_FORMAT = 'player_{index}'
MAX_CYCLES = 1000


class MeltingPotWrapper(MultiAgentEnv):
    """An adapter between Melting Pot substrates and MultiAgentEnv."""

    def __init__(self, env_config, env_name, render_mode, max_cycles=MAX_CYCLES):
        self.env_config = config_dict.ConfigDict(env_config)
        self.env_name = env_name
        self.render_mode = render_mode
        self.max_cycles = max_cycles

        # Build the environment
        self._env = substrate.build(
            env_name if env_name else self.env_config,
            roles=self.env_config.default_player_roles
        )
        self._num_players = len(self._env.observation_spec())

        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]

        observation_space = utils.remove_world_observations_from_space(
            utils.spec_to_space(self._env.observation_spec()[0])
        )
        self.observation_space = functools.lru_cache(maxsize=None)(lambda agent_id: observation_space)
        self.share_observation_space = self.observation_space

        self.action_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: utils.spec_to_space(self._env.action_spec()[0])
        )

        self.state_space = utils.spec_to_space(
            self._env.observation_spec()[0]['WORLD.RGB']
        )

        self.episode_limit = max_cycles
        self.n_agents = self._num_players
        self.agents = []
        self.num_cycles = 0

    def step(self, actions):
        """Takes a step in the environment."""
        actions_list = [actions[agent] for agent in self.agents]
        timestep = self._env.step(actions_list)

        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if done:
            self.agents = []

        observations = utils.timestep_to_observations(timestep)
        return observations, rewards, dones, dones, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        timestep = self._env.observe()
        return utils.timestep_to_observations(timestep)

    def get_obs_agent(self, agent_id):
        """Returns observation for a specific agent."""
        index = int(agent_id.split('_')[1])
        timestep = self._env.observe()
        return utils.spec_to_space(timestep[index])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return utils.spec_to_space(self._env.observation_spec()[0]).shape

    def get_state(self):
        """Returns the global state."""
        return self._env.observation()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.state_space.shape

    def get_avail_actions(self):
        """Returns available actions for all agents."""
        return [self.get_avail_agent_actions(agent) for agent in self.agents]

    def get_avail_agent_actions(self, agent_id):
        """Returns available actions for a specific agent."""
        index = int(agent_id.split('_')[1])
        return list(range(self.action_space(agent_id).n))

    def get_total_actions(self):
        """Returns the total number of possible actions."""
        return self.action_space(self.possible_agents[0]).n

    def reset(self, seed=None, options=None):
        """Resets the environment and returns initial observations."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        return utils.timestep_to_observations(timestep), {}

    def render(self, mode=None, filename=None, screen_width=800, screen_height=600, fps=8):
        """Renders the environment."""
        mode = mode or self.render_mode
        rgb_arr = self._env.observation_spec()[0]['WORLD.RGB']

        if mode == 'human':
            obs_spec = self._env.observation_spec()[0]['WORLD.RGB']
            obs_shape = obs_spec.shape
            scale = min(screen_height // obs_shape[0], screen_width // obs_shape[1])

            if not pygame.get_init():
                pygame.init()
                self.game_display = pygame.display.set_mode(
                    (obs_shape[1] * scale, obs_shape[0] * scale))
                pygame.display.set_caption('Melting Pot Environment')
                self.clock = pygame.time.Clock()
            # Add rendering logic if needed
        elif mode == 'rgb_array':
            return rgb_arr
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Closes the environment."""
        self._env.close()

    def seed(self, seed=None):
        """Sets the seed for the environment."""
        self._env.seed(seed)

    def save_replay(self):
        """Saves a replay of the environment."""
        # Placeholder implementation if saving replay is supported
        pass

    def get_env_info(self):
        """Returns environment metadata."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        """Returns environment statistics."""
        return {}
