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
    # Obtain solver configuration and environment instance using the provided parameters and checkpoint paths
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)
    env_info = env_copy.get_env_info()

    # Load the frozen policy from the checkpoint using the solver configuration and environment info
    frozen_policy = load_frozen_policy(solver_config, checkpoint_paths, env_info)

    # Generate candidate contract parameters (linearly sampled between 0 and 1)
    num_candidates = params_dict.get('solver_samples', 10)
    candidate_contracts = np.linspace(0, 1, num_candidates)

    best_reward = -float('inf')
    best_contract = None

    # Evaluate each candidate contract by performing multiple rollouts
    num_rollouts = params_dict.get('solver_rollouts', 5)
    for contract in candidate_contracts:
        env_copy.contract = contract
        total_reward = 0.0
        for _ in range(num_rollouts):
            obs = env_copy.reset()
            done = False
            ep_reward = 0.0
            # Run one rollout until the episode ends
            while not done:
                action = frozen_policy.compute_action(obs)
                obs, reward, terminated, truncated, info = env_copy.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
        avg_reward = total_reward / num_rollouts
        logger.log_stat("solver_contract_reward", avg_reward, 0)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_contract = contract

    logger.log_stat("optimal_contract", best_contract, 0)
    print("Solver evaluated candidate contracts:", candidate_contracts)
    print("Corresponding average rewards:", best_reward)
    print("Optimal contract selected:", best_contract)

    time.sleep(50)  # wait for logger to finish writing
    return best_contract
