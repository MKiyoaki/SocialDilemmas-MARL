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

# Import a function to get a solver-specific configuration and a clone of the environment.
# This function should return a configuration and an environment copy configured for solver evaluation.
from utils.ray_config_utils import get_solver_config


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
    # Get solver-specific configuration and a copy of the environment.
    # get_solver_config should create an environment instance that can be used for evaluation.
    solver_config, env_copy = get_solver_config(params_dict, checkpoint_paths)

    # Number of candidate contract values to evaluate.
    num_candidates = params_dict.get('solver_samples', 10)
    # Define candidate contract values uniformly in the range [0, 1].
    candidate_contracts = np.linspace(0, 1, num_candidates)

    candidate_rewards = []

    for contract in candidate_contracts:
        # Set the candidate contract in the environment.
        env_copy.contract = contract  # Assume the environment supports a 'contract' attribute.
        num_rollouts = params_dict.get('solver_rollouts', 5)
        total_reward = 0.0

        for _ in range(num_rollouts):
            obs = env_copy.reset()
            done = False
            episode_reward = 0.0
            while not done:
                # Compute the action using the frozen policy.
                # This placeholder assumes the environment provides a method to compute action.
                # In practice, use the appropriate method from your frozen policy (loaded from checkpoint).
                action = env_copy.compute_action(obs)
                obs, reward, done, info = env_copy.step(action)
                episode_reward += reward
            total_reward += episode_reward

        avg_reward = total_reward / num_rollouts
        candidate_rewards.append(avg_reward)
        logger.log_stat("solver_contract_reward", avg_reward, 0)

    # Select the contract with the highest average reward.
    best_index = np.argmax(candidate_rewards)
    optimal_contract = candidate_contracts[best_index]
    logger.log_stat("optimal_contract", optimal_contract, 0)

    # Log detailed solver results.
    print("Solver evaluated candidate contracts:", candidate_contracts)
    print("Corresponding average rewards:", candidate_rewards)
    print("Optimal contract selected:", optimal_contract)

    # Wait a bit to ensure logging completes.
    time.sleep(5)

    return optimal_contract
