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
    Transfer function for Prisoner's Dilemma environment that averages payoffs
    between cooperators and defectors.

    When agents choose different actions (one cooperates, one defects),
    the transfer function ensures both receive the average of their original payoffs.

    Args:
        obs: Tensor of observations for each agent
        acts: Tensor of actions for each agent (0 = cooperate, 1 = defect)
        rews: Tensor of original rewards for each agent
        params: Contract parameter controlling whether to apply the averaging
        infos: Additional information (optional)

    Returns:
        Tensor of reward transfers that implement payoff averaging
    """
    # Convert inputs to tensors if they aren't already
    if not isinstance(acts, th.Tensor):
        acts = th.tensor(acts, dtype=th.float32)
    if not isinstance(rews, th.Tensor):
        rews = th.tensor(rews, dtype=th.float32)

    # Initialize transfer tensor
    transfers = th.zeros_like(rews)

    # For 2-player Prisoner's Dilemma
    if len(acts) == 2:
        # Case 1: Player 0 cooperates and Player 1 defects
        if acts[0] == 0 and acts[1] == 1:
            # Calculate average reward
            avg_reward = (rews[0] + rews[1]) / 2
            # Calculate transfers needed to achieve average for both
            transfers[0] = avg_reward - rews[0]  # Cooperator receives
            transfers[1] = avg_reward - rews[1]  # Defector pays

        # Case 2: Player 1 cooperates and Player 0 defects
        elif acts[0] == 1 and acts[1] == 0:
            # Calculate average reward
            avg_reward = (rews[0] + rews[1]) / 2
            # Calculate transfers needed to achieve average for both
            transfers[0] = avg_reward - rews[0]  # Defector pays
            transfers[1] = avg_reward - rews[1]  # Cooperator receives

        # Case 3: Both cooperate or both defect - no transfers
        else:
            transfers[0] = 0
            transfers[1] = 0

    # Verify zero-sum property
    assert abs(th.sum(transfers).item()) < 1e-6, "Transfers must sum to zero"

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
