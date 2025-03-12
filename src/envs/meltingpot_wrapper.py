import numpy as np

from shimmy import MeltingPotCompatibilityV0
from gymnasium.spaces import Tuple
from gymnasium import register, spaces, Env


class MeltingPotPettingZooWrapper(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, substrate_name, **kwargs):
        super().__init__()
        self._env = MeltingPotCompatibilityV0(substrate_name=substrate_name, **kwargs)
        self.n_agents = len(self._env.possible_agents)
        self.agents = self._env.possible_agents

        self.observation_space = Tuple([self._env.observation_space(a) for a in self.agents])
        self.action_space = Tuple([self._env.action_space(a) for a in self.agents])

        self.last_obs = None
        self._unwrapped = self

    @property
    def unwrapped(self):
        return self._unwrapped

    def reset(self, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed, **kwargs)
        obs = tuple(obs[a] for a in self.agents)
        self.last_obs = obs
        return obs, info

    def step(self, actions):
        actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
        observations, rewards, dones, truncated, infos = self._env.step(actions_dict)

        obs = tuple(
            observations.get(
                agent,
                np.zeros(
                    self.observation_space.spaces[i].shape,
                    dtype=self.observation_space.spaces[i].dtype
                )
            )
            for i, agent in enumerate(self.agents)
        )
        rewards = [rewards.get(agent, 0) for agent in self.agents]
        done = all(dones.values())
        truncated = all(truncated.values())
        info = {
            f"{agent}_{key}": value
            for agent, agent_info in infos.items() for key, value in agent_info.items()
        }

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


def register_meltingpot():
    substrates = [
        "clean_up",
        "coins",
        "pure_coordination_in_the_matrix__repeated",
        "prisoners_dilemma_in_the_matrix__repeated",
        "stag_hunt_in_the_matrix__repeated",
        "chicken_in_the_matrix__repeated"
    ]
    for substrate_name in substrates:
        gymkey = f"pz-meltingpot-{substrate_name.replace('_', '-')}"
        register(
            gymkey,
            entry_point="envs.meltingpot_wrapper:MeltingPotPettingZooWrapper",
            kwargs={"substrate_name": substrate_name},
        )

register_meltingpot()