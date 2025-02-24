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
    Default reward transferring function: scales the reward by multiplying with (1 + theta)
    and adds theta for each agent.

    This function is designed for multiple agents. It assumes that 'rews' is a tensor of shape [B, T, n_agents].
    If 'params' is a dict, it uses params["a{i}"][0] as the theta for agent i; otherwise, it applies the same theta to all agents.

    Args:
        obs: Observation information.
        acts: Action information.
        rews: Original reward information, as a tensor of shape [B, T, n_agents] (or [B, T, 1] for common reward).
        params: Contract parameter for the transfer; either a float or a dict mapping agent keys to a 1-element list.
        infos: Additional information (optional).

    Returns:
        A tensor of adjusted reward transfers with shape [B, T, n_agents].
    """
    # If rews is a common reward tensor with shape [B, T, 1], expand it to [B, T, n_agents]
    if rews.size(-1) == 1:
        n_agents = 1  # Actually, common reward is shared; here we keep it as is.
        # For common reward, we apply the transformation and then (if needed) broadcast.
        adjusted = rews * (1 + params) + params if not isinstance(params, dict) else rews * (1 + params["a0"][0]) + params["a0"][0]
        return adjusted
    else:
        B, T, n_agents = rews.shape
        adjusted_list = []
        for i in range(n_agents):
            if isinstance(params, dict):
                theta = params.get(f"a{i}", [0.0])[0]
            else:
                theta = params
            # Apply the transformation element-wise
            adjusted = rews[..., i] * (1 + theta) + theta
            adjusted_list.append(adjusted)
        # Stack along the last dimension to get shape [B, T, n_agents]
        return th.stack(adjusted_list, dim=-1)

def pd_transfer_function(obs, acts, rews, params, infos=None):
    """
    Reward transferring function for the Prisoners Dilemma in the Matrix.

    Each agent's action: 0 means cooperate, 1 means defect.
    The parameter 'params' represents the contract parameter (theta) for the transfer.
    It can be either a float (in which case the same theta is used for both agents)
    or a dict with keys 'a0' and 'a1'.

    If agent0 defects while agent1 cooperates, then agent0 pays theta (negative transfer)
    and agent1 receives theta (positive transfer). Similarly, if agent0 cooperates and agent1 defects,
    then agent0 receives theta and agent1 pays theta. Otherwise, no transfer occurs.

    Args:
        obs: Observation information.
        acts: Action information. Expected shape is [B, T, 2, 1].
        rews: Original reward information as a tensor.
        params: Contract parameter for the transfer (float or dict).
        infos: Additional information (optional).

    Returns:
        A tensor of reward transfers with shape [B, T, 2].
    """
    # Squeeze the last dimension to get shape [B, T, 2]
    acts = acts.squeeze(-1)

    # Extract actions for agent0 and agent1: both have shape [B, T]
    a0 = acts[..., 0]
    a1 = acts[..., 1]

    # Determine theta values
    if isinstance(params, dict):
        theta0 = params["a0"][0] if "a0" in params else 0.0
        theta1 = params["a1"][0] if "a1" in params else 0.0
    else:
        theta0 = theta1 = params

    # Create condition masks:
    cond0 = (a0 == 1) & (a1 == 0)  # agent0 defects and agent1 cooperates
    cond1 = (a0 == 0) & (a1 == 1)  # agent0 cooperates and agent1 defects

    # Compute transfers for each agent
    transfer_a0 = th.where(cond0, -theta0, th.zeros_like(a0))
    transfer_a0 = th.where(cond1, theta1, transfer_a0)

    transfer_a1 = th.where(cond0, theta0, th.zeros_like(a1))
    transfer_a1 = th.where(cond1, -theta1, transfer_a1)

    # Stack the transfers to form a tensor of shape [B, T, 2]
    transfers = th.stack([transfer_a0, transfer_a1], dim=-1)
    return transfers


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
