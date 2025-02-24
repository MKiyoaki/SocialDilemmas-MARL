import gymnasium as gym
import numpy as np
import torch as th


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


def pd_transfer_function(obs, acts, rews, params, infos=None):
    """
    Reward transferring function for the Prisoners Dilemma in the Matrix.

    Each agent's action: 0 means cooperate, 1 means defect.
    The parameter 'params' represents the contract parameter (theta) for each agent.

    If agent i defects while the other agent cooperates,
    then agent i pays theta (negative transfer) and agent j receives theta (positive transfer).
    Otherwise, no transfer occurs.

    Args:
        obs: Observation information.
        acts: Action information. Can be a dict or a tensor.
        rews: Original reward information.
        params: Contract parameter for the transfer.
        infos: Additional information (optional).

    Returns:
        A dict of reward transfers for each agent.
    """
    # If acts is a tensor, convert it into a dictionary with keys 'a0' and 'a1'
    # TODO FIX this
    print("DEBUG: " + acts)

    if isinstance(acts, th.Tensor):
        # Assume acts is of shape (num_agents,) or (1, num_agents); flatten it to a list
        acts = acts.view(-1).tolist()
        acts = {f"a{i}": act for i, act in enumerate(acts)}

    transfers = {}
    agents = list(acts.keys())
    # Assume only two agents are present
    if len(agents) != 2:
        raise ValueError("Prisoners Dilemma Transfer function supports only 2 agents.")
    a0, a1 = agents[0], agents[1]

    # Initialize transfers
    transfers[a0] = 0
    transfers[a1] = 0

    # If a0 defects (1) and a1 cooperates (0), a0 pays theta and a1 receives theta.
    if acts[a0] == 1 and acts[a1] == 0:
        transfers[a0] = -params[a0][0]
        transfers[a1] = params[a0][0]
    # If a0 cooperates (0) and a1 defects (1), a0 receives theta and a1 pays theta.
    elif acts[a0] == 0 and acts[a1] == 1:
        transfers[a0] = params[a1][0]
        transfers[a1] = -params[a1][0]
    # If both choose the same action, no transfer occurs.
    else:
        transfers[a0] = 0
        transfers[a1] = 0

    return transfers


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
    elif name == "pd_transfer_function":
        return pd_transfer_function
    # elif ...
    else:
        raise ValueError(f"Unknown transfer function: {name}")
