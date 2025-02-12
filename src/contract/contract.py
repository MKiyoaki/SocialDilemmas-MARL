import gymnasium as gym
import numpy as np

class Contract:
    def __init__(self, contract_space, default_contract, num_agents, features_compute=None):
        self.contract_space = contract_space
        self.default_contract = default_contract
        self.features_compute = features_compute
        self.num_agents = num_agents

    def compute_transfer(self, obs, acts, params, infos=None):
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
        """
        return self.transfer_function(obs, acts, rews, params, infos)
