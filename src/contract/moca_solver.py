"""
moca_solver.py

This module implements the MOCA solver that uses a frozen policy (obtained from training)
to evaluate and determine the optimal contract parameter. It performs multiple rollouts
with different contract values and selects the contract with the best performance.
"""

import time
import copy
import torch as th
import numpy as np

from src.utils.config_utils import get_solver_config_from_params
from src.utils.model_utils import load_frozen_policy


def run_solver(params_dict, checkpoint_paths, logger):
    """
    Run the MOCA solver to determine the optimal contract parameter using a frozen policy.

    Parameters:
        params_dict (dict): Experiment parameter dictionary.
        checkpoint_paths (list): List of checkpoint paths from training.
        logger: Logger for recording statistics.

    Returns:
        optimal_contract: The contract value that yields the best performance.
    """
    # Obtain a solver-specific configuration and an environment instance for evaluation.
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)

    # Load the frozen policy from the saved checkpoints.
    store_path = params_dict.get("store_path", None)
    frozen_policy = load_frozen_policy(store_path, checkpoint_paths)

    num_candidates = params_dict.get('solver_samples', 10)
    candidate_contracts = np.linspace(0, 1, num_candidates)
    candidate_rewards = []

    for contract in candidate_contracts:
        # Set the candidate contract in the environment; assume env_copy has a contract attribute.
        env_copy.contract = contract
        num_rollouts = params_dict.get('solver_rollouts', 5)
        total_reward = 0.0

        for _ in range(num_rollouts):
            obs = env_copy.reset()
            done = False
            episode_reward = 0.0
            while not done:
                # Use the frozen policy to compute action based on the observation.
                action = frozen_policy.compute_action(obs)
                obs, reward, done, info = env_copy.step(action)
                episode_reward += reward
            total_reward += episode_reward

        avg_reward = total_reward / num_rollouts
        candidate_rewards.append(avg_reward)
        logger.log_stat("solver_contract_reward", avg_reward, 0)

    best_index = np.argmax(candidate_rewards)
    optimal_contract = candidate_contracts[best_index]
    logger.log_stat("optimal_contract", optimal_contract, 0)

    print("Solver evaluated candidate contracts:", candidate_contracts)
    print("Corresponding average rewards:", candidate_rewards)
    print("Optimal contract selected:", optimal_contract)

    time.sleep(5)
    return optimal_contract
