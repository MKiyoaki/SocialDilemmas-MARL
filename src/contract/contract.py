import torch as th
import numpy as np
import gymnasium as gym


class Contract:
    def __init__(self, contract_space, default_contract, num_agents, features_compute=None):
        self.contract_space = contract_space
        self.default_contract = default_contract
        self.features_compute = features_compute  # 用于从 obs 提取特征
        self.num_agents = num_agents

    def compute_transfer(self, obs, acts, rews, params, infos=None):
        raise NotImplementedError("compute_transfer must be implemented by subclass.")


class GeneralContract(Contract):
    """
    GeneralContract redefines the contract as a function mapping from observable features to a reward transfer vector theta.
    The transfer function returns a tensor theta of shape [B, T, num_agents].
    We then normalize theta to ensure zero-sum across agents.
    """

    def __init__(self, num_agents, contract_type, params_range, transfer_function, features_compute=None):
        """
        Args:
            num_agents (int): Number of agents.
            contract_type (str): Type of the environment.
            params_range (tuple): Parameter range (used for sampling initial parameters).
            transfer_function (function): A function mapping features, acts, rews and contract parameters to a contract vector theta.
                                          Expected signature: transfer_function(features, acts, rews, params, infos, num_agents)
            features_compute (function, optional): A function to extract features from observations.
        """
        super().__init__(gym.spaces.Box(shape=(1,), low=params_range[0], high=params_range[1]),
                         np.array([0.0]), num_agents, features_compute=features_compute)
        self.contract_type = contract_type
        self.transfer_function = transfer_function
        self.contract_space_low = params_range[0]
        self.contract_space_high = params_range[1]

    def compute_transfer(self, obs, acts, rews, params, infos=None):
        """
        Compute reward transferring based on the state-dependent contract function.

        Steps:
         1. If features_compute is provided, extract features from obs; otherwise use obs.
         2. Compute contract vector theta = transfer_function(features, acts, rews, params, infos, num_agents).
         3. Normalize theta along the agent dimension to ensure zero-sum.
         4. Adjust the original rewards using theta (here simply add theta to the rewards).

        Args:
            obs: Observation information, expected shape [B, T_obs, obs_dim] or [B, T_obs, num_agents, d].
            acts: Action information.
            rews: Original reward information as a tensor of shape [B, T_rews, num_agents].
            params: Contract parameters (scalar or tensor) used in transfer_function.
            infos: Additional information (optional).

        Returns:
            A tensor of adjusted rewards with the same shape as rews.
        """
        # Step 1: Extract features from obs if possible
        if self.features_compute is not None:
            features = self.features_compute(obs)
        else:
            if not isinstance(obs, th.Tensor):
                features = th.tensor(obs, dtype=th.float32)
            else:
                features = obs

        # Step 2: Compute contract vector theta using the transfer function.
        theta = self.transfer_function(features, acts, rews, params, infos, num_agents=self.num_agents)

        # Step 3: Normalize theta so that sum along the agent dimension equals zero.
        theta = theta - theta.mean(dim=-1, keepdim=True)

        # Adjust time dimension: if theta has one more timestep than rews, slice the last timestep.
        if theta.shape[1] != rews.shape[1]:
            if theta.shape[1] == rews.shape[1] + 1:
                theta = theta[:, :-1, :]
            else:
                raise ValueError("Time dimension mismatch between computed theta and rewards.")

        # Step 4: Adjust rewards by adding theta.
        adjusted_rews = rews + theta
        return adjusted_rews


def default_transfer_function(features, acts, rews, params, infos=None, num_agents=2):
    """
    A basic default state-dependent contract function for debugging.

    If features have shape [B, T, num_agents, d], this function ignores the feature content
    and returns a constant contract vector for each sample.

    When params is a scalar (float or int), it generates a constant vector using a linear space
    from -0.5 to 0.5, multiplied by params. The resulting vector is normalized to have zero sum.

    Args:
        features: A tensor of shape [B, T, num_agents, d] (e.g., [64, 101, 2, d]).
        acts: Not used in this function.
        rews: Not used in this function.
        params: Contract parameter (a scalar) used to scale the contract vector.
        infos: Additional information (unused).
        num_agents: Number of agents.

    Returns:
        A tensor theta of shape [B, T, num_agents], where the sum over the agent dimension is zero.
    """
    B, T, N, d = features.shape
    if N != num_agents:
        raise ValueError("Mismatch: features has {} agents but num_agents is {}".format(N, num_agents))

    if isinstance(params, (float, int)):
        theta_const = th.linspace(-0.5, 0.5, steps=num_agents, device=features.device)
        theta = theta_const.unsqueeze(0).unsqueeze(0).expand(B, T, num_agents) * params
    else:
        # For non-scalar params, perform a linear mapping using global features (averaged over agents)
        global_features = features.mean(dim=2)  # shape [B, T, d]
        theta = th.matmul(global_features, params)  # Expecting params shape [d, num_agents]
    theta = theta - theta.mean(dim=-1, keepdim=True)
    return theta


def pd_transfer_function(features, acts, rews, params, infos=None, num_agents=2):
    """
    Reward transferring function for the Prisoners Dilemma.
    This implementation remains state-independent.

    Args:
        features: Not used for PD, can be ignored.
        acts: Action tensor, expected shape [B, T, 2, ...]. Squeeze the last dimension if needed.
        rews: Not used in the computation.
        params: Contract parameter (either a float or a dict with keys for each agent).
        infos: Additional information (optional).
        num_agents: Number of agents (should be 2 for PD).

    Returns:
        A tensor theta of reward transfers with shape [B, T, 2].
    """
    acts = acts.squeeze(-1)  # Ensure shape [B, T, 2]
    a0 = acts[..., 0]
    a1 = acts[..., 1]

    if isinstance(params, dict):
        theta0 = params.get("a0", [0.0])[0]
        theta1 = params.get("a1", [0.0])[0]
    else:
        theta0 = theta1 = params

    cond0 = (a0 == 1) & (a1 == 0)
    cond1 = (a0 == 0) & (a1 == 1)

    transfer_a0 = th.where(cond0, -theta0, th.zeros_like(a0))
    transfer_a0 = th.where(cond1, theta1, transfer_a0)
    transfer_a1 = th.where(cond0, theta0, th.zeros_like(a1))
    transfer_a1 = th.where(cond1, -theta1, transfer_a1)

    theta = th.stack([transfer_a0, transfer_a1], dim=-1)  # shape [B, T, 2]
    # Ensure zero-sum normalization (though PD logic should already yield zero sum)
    theta = theta - theta.mean(dim=-1, keepdim=True)
    return theta


def get_transfer_function(name: str):
    """
    Return the corresponding reward transfer function based on the given name.
    """
    if name == "default_transfer_function":
        return default_transfer_function
    elif name == "pd_transfer_function":
        return pd_transfer_function
    else:
        raise ValueError(f"Unknown transfer function: {name}")
