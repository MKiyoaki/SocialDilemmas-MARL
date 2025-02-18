import time
import copy
import torch as th
import numpy as np

from src.utils.config_utils import get_solver_config_from_params
from src.utils.model_utils import load_frozen_policy
from src.contract.contract import GeneralContract, default_transfer_function


def run_solver(params_dict, checkpoint_paths, logger):
    """
    Obtain solver configuration and environment instance using provided parameters and checkpoint paths

    Args:
        params_dict (dict): Dictionary of solver parameters.
        checkpoint_paths (list): List of paths to checkpoint files.
        logger (Logger): Logger instance.
    Returns:
        Optimal contract.
    """
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)
    env_info = env_copy.get_env_info()

    # Load the frozen policy from the checkpoint using the solver configuration and environment info
    frozen_policy = load_frozen_policy(solver_config, checkpoint_paths, env_info)

    # Create a GeneralContract instance to be used for reward adjustment.
    # This instance is required for computing adjusted_reward.
    contract_instance = GeneralContract(
        num_agents=env_info["n_agents"],
        contract_type="general",
        params_range=(0.0, 1.0),
        transfer_function=default_transfer_function
    )

    candidate_contracts = np.array(params_dict["candidate_contracts"], dtype=float)
    logger.console_logger.info("Using candidate contracts: {}".format(candidate_contracts))

    best_reward = -float('inf')
    best_contract = None

    # Evaluate each candidate contract by performing multiple rollouts
    num_rollouts = params_dict.get('solver_rollouts', 5)
    for contract in candidate_contracts:
        env_copy.contract = contract  # Set the environment's contract parameter
        total_reward = 0.0
        for _ in range(num_rollouts):
            obs = env_copy.reset()
            done = False
            ep_reward = 0.0
            # Run a rollout until the episode ends
            while not done:
                action = frozen_policy.compute_action(obs)
                obs, reward, terminated, truncated, info = env_copy.step(action)
                done = terminated or truncated
                # Use the contract_instance to compute adjusted_reward
                adjusted_reward = contract_instance.compute_transfer(obs, action, reward, contract, info)
                ep_reward += adjusted_reward
            total_reward += ep_reward
        avg_reward = total_reward / num_rollouts
        logger.log_stat("solver_contract_reward", avg_reward, 0)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_contract = contract

    logger.log_stat("solver_optimal_contract", best_contract, 0)
    print("Solver evaluated candidate contracts:", candidate_contracts)
    print("Corresponding average rewards:", best_reward)
    print("Optimal contract selected:", best_contract)

    return best_contract
