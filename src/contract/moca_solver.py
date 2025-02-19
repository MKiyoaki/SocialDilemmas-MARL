import time
import copy
import torch as th
import numpy as np

from src.utils.config_utils import get_solver_config_from_params
from src.utils.model_utils import load_frozen_policy
from src.contract.contract import GeneralContract, default_transfer_function

def run_solver(params_dict, checkpoint_paths, logger):
    """
    Obtain solver configuration and environment instance using provided parameters and checkpoint paths.

    Args:
        params_dict (dict): Dictionary of solver parameters.
        checkpoint_paths (list): List of paths to checkpoint files.
        logger (Logger): Logger instance.
    Returns:
        Optimal contract.
    """
    # Get solver config and environment copy
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)
    env_info = env_copy.get_env_info()

    # Load the frozen policy from the checkpoint using the solver configuration and environment info.
    # Assumes frozen_policy has a method compute_single_action(obs, policy_id).
    frozen_policy = load_frozen_policy(solver_config, checkpoint_paths, env_info)

    # Create a GeneralContract instance for reward adjustment.
    # This instance is used to compute the adjusted reward if needed.
    contract_instance = GeneralContract(
        num_agents=env_info["n_agents"],
        contract_type="general",
        params_range=(0.0, 1.0),
        transfer_function=default_transfer_function
    )

    # Retrieve candidate contracts from parameters
    candidate_contracts = np.array(params_dict["candidate_contracts"], dtype=float)
    logger.console_logger.info("Using candidate contracts: {}".format(candidate_contracts))

    best_reward = -float('inf')
    best_contract = None

    # Get the number of rollouts per candidate
    num_rollouts = params_dict.get('solver_rollouts', 5)
    for contract in candidate_contracts:
        # Set the environment's contract parameter
        env_copy.contract = contract
        total_reward = 0.0
        for _ in range(num_rollouts):
            # Reset the environment; assume obs is a dictionary: {agent_id: obs_data, ...}
            obs = env_copy.reset()
            done = False
            ep_reward = 0.0

            # Check if obs is a dict; if so, set active_agents to all keys
            if isinstance(obs, dict):
                active_agents = list(obs.keys())
            else:
                active_agents = None

            while not done:
                if active_agents is not None:
                    act_dict = {}
                    # Compute actions separately for each agent
                    for agent_id in active_agents:
                        # Use shared policy if specified, else use each agent's own policy
                        if params_dict.get('shared_policy', True):
                            action = frozen_policy.compute_single_action(obs[agent_id], policy_id='policy')
                        else:
                            action = frozen_policy.compute_single_action(obs[agent_id], policy_id=agent_id)
                        act_dict[agent_id] = action
                    # Step the environment with the action dictionary
                    next_obs, reward, dones, info = env_copy.step(act_dict)
                    # Check for overall termination
                    if dones.get('__all__', False):
                        done = True
                    else:
                        # Remove agents that are done from active_agents
                        active_agents = [agent_id for agent_id in active_agents if not dones.get(agent_id, False)]
                    # Sum rewards from all agents (assumes reward is a dict)
                    ep_reward += sum(reward.values())
                    obs = next_obs
                else:
                    # If obs is not a dict, fallback to a unified action computation
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

    return best_contract
