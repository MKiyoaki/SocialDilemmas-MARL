import warnings
from collections.abc import Iterable
import numpy as np
import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit

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
        contract=None,  # Optional contract instance for MOCA; default is None
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

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except Exception:
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
        self.contract = contract
        if self.contract is not None:
            self.contract_low = self.contract.contract_space_low
            self.contract_high = self.contract.contract_space_high
            # Initialize the contract params
            self.current_contract = getattr(kwargs.get('args', {}), 'chosen_contract', None)
            if self.current_contract is None:
                self.current_contract = np.random.uniform(low=self.contract_low, high=self.contract_high)
            if isinstance(self._env.observation_space, gym.spaces.Box):
                new_low = np.concatenate(
                    (self._env.observation_space_low, np.array([self.contract_low]), np.array([0]))
                )
                new_high = np.concatenate(
                    (self._env.observation_space_high, np.array([self.contract_high]), np.array([3]))
                )
                self.observation_space = gym.spaces.Box(
                    low=new_low, high=new_high, dtype=self._env.observation_space.dtype
                )
            else:
                self.observation_space = self._env.observation_space
        else:
            self.observation_space = self._env.observation_space
        # -----------------------------------------------------------

    def update_contract(self, new_contract):
        """
        Update the environment's current contract parameter.
        This method is intended to be called by the solver after computing the optimal contract.
        """
        if self.contract is not None:
            self.current_contract = new_contract

    def _pad_observation(self, obs):
        """Pad each observation to match the longest observation space."""
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def reset(self, seed=None, options=None):
        """Reset the environment. Extend observations with the current contract parameter if applicable."""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        if self.contract is not None:
            cp = self.current_contract
            self._obs = [np.concatenate((o, np.array([cp]), np.array([0]))) for o in self._obs]
        return self._obs, info

    def step(self, actions):
        """
        Execute one step in the environment.
        If a contract is provided, adjust rewards using contract.compute_transfer and extend observations.
        """
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )
        if isinstance(done, Iterable):
            done = all(done)

        if self.contract is not None:
            agent_names = ["a" + str(i) for i in range(self.n_agents)]
            if isinstance(reward, list):
                base_rewards = {agent_names[i]: reward[i] for i in range(self.n_agents)}
            else:
                base_rewards = {agent: reward for agent in agent_names}
            adjusted_rewards = self.contract.compute_transfer(self._obs, actions, base_rewards, self.current_contract, self._info)
            new_reward = [adjusted_rewards.get(agent, base_rewards.get(agent, 0)) for agent in agent_names]
            new_obs = [np.concatenate((o, np.array([self.current_contract]), np.array([0]))) for o in self._obs]
        else:
            new_reward = reward
            new_obs = self._obs

        return new_obs, new_reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations."""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns the observation for a specific agent."""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the flattened observation size."""
        base_size = flatdim(self.longest_observation_space)
        return base_size + (2 if self.contract is not None else 0)

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
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for an agent."""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of possible actions."""
        return flatdim(self.longest_action_space)

    def reset_env(self, seed=None, options=None):
        """An alias for reset()."""
        return self.reset(seed=seed, options=options)

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
