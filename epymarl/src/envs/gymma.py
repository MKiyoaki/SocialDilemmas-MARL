from collections.abc import Iterable
import warnings
import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa

try:
    from .pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )


class GymmaWrapper(MultiAgentEnv):
    def __init__(
            self,
            key,
            time_limit,
            pretrained_wrapper,
            seed,
            common_reward,
            reward_scalarisation,
            **kwargs,
    ):
        # Create the base gym environment
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        # MOCA flag: if True, enable MOCA functionality
        self.moca = kwargs.get("moca", False)

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

        # ------------------ MOCA Contract Support ------------------
        # If MOCA is enabled, expect a contract function/object to be provided via kwargs.
        # The contract should be callable and return a tuple (contract_value, contract_state)
        # and also provide a compute_transfer() method for reward adjustment.
        if self.moca:
            self.contract = kwargs.get("contract", None)
            if self.contract is None:
                raise ValueError("MOCA is enabled but no contract function/object provided.")
            # Initialize dynamic contract info
            self.current_contract = None
            self.contract_state = None
        # -----------------------------------------------------------

    def update_contract(self, new_contract_param):
        """
        Update the environment's current contract parameter.
        This method allows external modules (e.g., the solver) to update the contract parameter.
        In this simple implementation, we just set current_contract to new_contract_param.
        """
        if self.moca:
            self.current_contract = new_contract_param
        else:
            # If MOCA is not enabled, do nothing.
            pass

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns observations, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)

        if self.moca:
            # MOCA branch: update dynamic contract info based on current observation and info.
            # Call the contract function/object to compute (contract_value, contract_state)
            cp, cs = self.contract(self._obs, self._info)  # contract should be callable: (obs, info) -> (cp, cs)
            self.current_contract = cp
            self.contract_state = cs
            # Adjust rewards using contract.compute_transfer()
            transfers = self.contract.compute_transfer(self._obs, actions, reward, self._info)
            # Update reward for each agent; assume reward is a list/iterable.
            if isinstance(reward, Iterable):
                agent_names = ["a" + str(i) for i in range(self.n_agents)]
                reward = [reward[i] + transfers.get(agent_names[i], 0) for i in range(self.n_agents)]
            # Extend each observation with the current contract value and contract state
            self._obs = [np.concatenate((o, np.array([cp]), np.array([cs]))) for o in self._obs]

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for a specific agent"""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the flattened observation size. If MOCA is enabled, additional dimensions are added."""
        base_size = flatdim(self.longest_observation_space)
        if self.moca:
            return base_size + 2  # One for contract value and one for contract state
        else:
            return base_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the state size."""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for a specific agent"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info.

        If MOCA is enabled, compute and append contract info to the observations.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        if self.moca:
            # Compute dynamic contract info from observations and info.
            cp, cs = self.contract(self._obs, info)  # contract function: (obs, info) -> (contract_value, contract_state)
            self.current_contract = cp
            self.contract_state = cs
            # Append contract info to each observation.
            self._obs = [np.concatenate((o, np.array([cp]), np.array([cs]))) for o in self._obs]
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
