from gymnasium import Wrapper, spaces
import numpy as np
from pettingzoo.utils import BaseParallelWrapper


class FlattenObservation(Wrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, actions):
        obs, rew, done, truncated, info = self.env.step(actions)
        return self._flatten_obs(obs), rew, done, truncated, info

    def _flatten_obs(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class ParallelFlattenObservation(BaseParallelWrapper):
    """Observation wrapper that flattens the observation of individual agents in a parallel environment."""

    def __init__(self, env):
        super().__init__(env)
        self.obs_space = {

            agent: spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(spaces.flatdim(env.observation_space(agent)),),
                dtype=np.float32
            )
            for agent in env.possible_agents
        }
        self.obs_space = spaces.Tuple(tuple(self.obs_space.values()))


    def reset(self, seed=None, options=None):
        observations = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(observations)

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        return (
            self._flatten_obs(observations),
            rewards,
            dones,
            infos,
        )

    def _flatten_obs(self, observations):
        return {
            agent: spaces.flatten(self.env.observation_space(agent), obs)
            for agent, obs in observations.items()
        }
