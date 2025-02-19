import gymnasium as gym
import numpy as np


class Contract:
    def __init__(self, contract_space, default_contract, num_agents, features_compute=None):
        self.contract_space = contract_space
        self.default_contract = default_contract
        self.features_compute = features_compute
        self.num_agents = num_agents

    def compute_transfer(self, obs, acts, rews, params, infos=None):
        raise NotImplementedError


class GeneralContract(Contract):
    """

    """
    def __init__(self,  num_agents, contract_type, params_range, transfer_function):
        """
        Args:
            num_agents (int): Number of the agents
            contract_type (str): Type of the environment
            params_range (tuple): Param scope
            transfer_function (function): Reward transferring function
        """
        super().__init__(gym.spaces.Box(shape=(1,), low=params_range[0], high=params_range[1]), np.array([0.0]), num_agents)
        self.contract_type = contract_type
        self.transfer_function = transfer_function
        self.contract_space_low = params_range[0]
        self.contract_space_high = params_range[1]

    def compute_transfer(self, obs, acts, rews, params, infos=None):
        """
        Compute reward transferring by the function transfer_function

        Args:
            obs: Observation information
            acts: Action information
            rews: Original reward information
            params: Parameter for the contract
            infos: Extra infos

        Returns:
            Reward value after adjusting function is applied.
        """
        return self.transfer_function(obs, acts, rews, params, infos)


def default_transfer_function(obs, acts, rews, params, infos=None):
    """
    Default reward transferring function：Scaling the reward by multiplying with (1 + params).

    Args:
        obs: Observation information
        acts: Action information
        rews: Original reward information
        params: Parameter for the contract
        infos: Extra infos

    Returns:
        Reward value after adjusting function is applied.
    """
    return rews * (1 + params) + params


# In src/contract/contract.py

def get_transfer_function(name: str):
    """
    Return the corresponding reward transfer function based on the given name.

    Args:
        name (str): The name of the transfer function.

    Returns:
        function: The corresponding transfer function.

    Raises:
        ValueError: If the name does not match any available function.
    """
    if name == "default_transfer_function":
        return default_transfer_function
    # 这里可以继续添加更多匹配项，例如：
    # elif name == "another_transfer_function":
    #     return another_transfer_function
    else:
        raise ValueError(f"Unknown transfer function: {name}")
