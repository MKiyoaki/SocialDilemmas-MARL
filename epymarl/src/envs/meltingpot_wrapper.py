from shimmy import MeltingPotCompatibilityV0
import numpy as np
from gymnasium.spaces import Tuple
from gymnasium import register, spaces, Env
from pettingzoo.utils.env import ParallelEnv

class MeltingPotPettingZooWrapper(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, substrate_name, **kwargs):
        super().__init__()
        self._env = MeltingPotCompatibilityV0(substrate_name=substrate_name, **kwargs)
        self.n_agents = len(self._env.possible_agents)
        self.agents = self._env.possible_agents

        # Define observation and action spaces
        self.observation_space = Tuple([self._env.observation_space(a) for a in self.agents])
        self.action_space = Tuple([self._env.action_space(a) for a in self.agents])

        self.last_obs = None

        # print("Observation space for each agent:")
        # for agent in self.agents:
        #     print(f"{agent}: {self._env.observation_space(agent)}")

    def reset(self, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        obs = tuple(obs[a] for a in self.agents)
        self.last_obs = obs
        return obs, info

    def step(self, actions):
        actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
        observations, rewards, dones, truncated, infos = self._env.step(actions_dict)

        obs = tuple(observations.get(agent, np.zeros_like(self.observation_space.spaces[i].shape))
                    for i, agent in enumerate(self.agents))
        rewards = [rewards.get(agent, 0) for agent in self.agents]
        done = all(dones.values())
        truncated = all(truncated.values())
        info = {f"{agent}_{key}": value for agent, agent_info in infos.items() for key, value in agent_info.items()}

        if done:
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs

        return obs, rewards, done, truncated, info

    def render(self, mode="human"):
        self.render_mode = mode
        return self._env.render()

    def close(self):
        return self._env.close()


class MeltingPotGymWrapper(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, substrate_name, **kwargs):
        super().__init__()
        self._pz_env = MeltingPotPettingZooWrapper(substrate_name=substrate_name, **kwargs)

        self.action_space = spaces.Tuple(self._pz_env.action_space.spaces)
        self.observation_space = spaces.Tuple(self._pz_env.observation_space.spaces)

        self.n_agents = self._pz_env.n_agents

    def reset(self, seed=None, **kwargs):
        obs, info = self._pz_env.reset(seed=seed, **kwargs)
        return obs, info

    def step(self, action):
        obs, rewards, done, truncated, info = self._pz_env.step(action)
        return obs, rewards, done, truncated, info

    def render(self, mode="human"):
        return self._pz_env.render(mode)

    def close(self):
        self._pz_env.close()


def register_meltingpot():
    substrates = [
        "prisoners_dilemma_in_the_matrix__arena",
        "prisoners_dilemma_in_the_matrix__repeated",
        "clean_up",
        "coins",
        "pure_coordination_in_the_matrix__repeated"
    ]
    for substrate_name in substrates:
        gymkey = f"pz-meltingpot-{substrate_name.replace('_', '-')}"
        register(
            gymkey,
            entry_point="envs.meltingpot_wrapper:MeltingPotGymWrapper",
            kwargs={"substrate_name": substrate_name},
        )

register_meltingpot()