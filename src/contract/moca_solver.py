import copy
import numpy as np
import time
import torch as th
from src.utils.config_utils import get_solver_config_from_params
from src.utils.model_utils import load_frozen_policy
from src.contract.contract import GeneralContract, default_transfer_function


def run_solver(params_dict, checkpoint_paths, logger):
    """
    Run the solver to determine the optimal contract.
    After computing the optimal contract, update the environment via update_contract().
    """
    # Get solver config and environment copy
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)
    env_info = env_copy.get_env_info()

    # Load the frozen policy from the checkpoint
    frozen_policy = load_frozen_policy(solver_config, checkpoint_paths, env_info)

    # Create a GeneralContract instance for evaluation
    contract_instance = GeneralContract(
        num_agents=env_info["n_agents"],
        contract_type="general",
        params_range=(0.0, 1.0),
        transfer_function=default_transfer_function
    )

    # Update contract_instance with proper low/high fields (if needed)
    candidate_contracts = np.array(params_dict["candidate_contracts"], dtype=float)
    logger.console_logger.info("Using candidate contracts: {}".format(candidate_contracts))

    best_reward = -float('inf')
    best_contract = None

    num_rollouts = params_dict.get('solver_rollouts', 5)
    for contract in candidate_contracts:
        env_copy.update_contract(contract)
        total_reward = 0.0
        for _ in range(num_rollouts):
            obs = env_copy.reset()[0]
            done = False
            ep_reward = 0.0
            if isinstance(obs, dict):
                active_agents = list(obs.keys())
            else:
                active_agents = None

            while not done:
                if active_agents is not None:
                    act_dict = {}
                    for agent_id in active_agents:
                        if params_dict.get('shared_policy', True):
                            action = frozen_policy.compute_single_action(obs[agent_id], policy_id='policy')
                        else:
                            action = frozen_policy.compute_single_action(obs[agent_id], policy_id=agent_id)
                        act_dict[agent_id] = action
                    next_obs, reward, dones, info = env_copy.step(act_dict)
                    if dones.get('__all__', False):
                        done = True
                    else:
                        active_agents = [agent_id for agent_id in active_agents if not dones.get(agent_id, False)]
                    ep_reward += sum(reward.values()) if isinstance(reward, dict) else reward
                    obs = next_obs
                else:
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

    logger.log_stat("solver_optimal_contract", best_contract, 0)

    # Update the contract parameter in the main environment
    env_copy.update_contract(best_contract)
    return best_contract
