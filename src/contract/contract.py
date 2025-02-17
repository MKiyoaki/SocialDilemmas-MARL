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
    def __init__(self, num_agents, contract_type, params_range, transfer_function):
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

    def compute_transfer(self, obs, acts, rews, params, infos=None):
        """
        Compute reward transferring by the function transfer_function

        Args:
            obs: 观察数据
            acts: 动作数据
            rews: 原始奖励数据（可以是 numpy 数组或其他数据类型）
            params: 契约参数（假定为数值型，或与 rews 可直接计算）
            infos: 可选的额外信息

        Returns:
            调整后的奖励数据
        """
        return self.transfer_function(obs, acts, rews, params, infos)


def default_transfer_function(obs, acts, rews, params, infos=None):
    """
    Default reward transferring function：Scaling the reward by multiplying with (1 + params).

    Args:
        obs: 观察数据
        acts: 动作数据
        rews: 原始奖励数据（可以是 numpy 数组或其他数据类型）
        params: 契约参数（假定为数值型，或与 rews 可直接计算）
        infos: 可选的额外信息

    Returns:
        调整后的奖励数据
    """
    return rews * (1 + params)
