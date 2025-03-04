import copy
import numpy as np
import time
import torch as th
from src.utils.config_utils import get_solver_config_from_params
from src.utils.model_utils import load_frozen_policy

def run_solver(params_dict, checkpoint_paths, logger):
    """
    Run the solver to determine the optimal contract parameters.
    Candidate contract parameters are obtained from the learner.
    The solver evaluates each candidate by performing multiple rollouts
    and then selects the candidate with the highest average reward.
    Finally, it updates the environment's contract parameters via update_contract().
    """
    # Get solver config and environment copy
    solver_config, env_copy = get_solver_config_from_params(params_dict, checkpoint_paths)
    env_info = env_copy.get_env_info()

    # Load the frozen policy from the checkpoint
    frozen_policy = load_frozen_policy(solver_config, checkpoint_paths, env_info)

    # Candidate contract parameters are provided by the learner via params_dict
    candidate_contracts = np.array(params_dict["candidate_contracts"], dtype=float)
    logger.console_logger.info("Using candidate contract parameters: {}".format(candidate_contracts))

    best_reward = -float('inf')
    best_contract_param = None

    num_rollouts = params_dict.get('solver_rollouts', 5)
    for candidate in candidate_contracts:
        # Update the environment's contract parameter (for state-independent contract, candidate is a scalar parameter)
        env_copy.update_contract(candidate)
        logger.console_logger.info("Testing candidate contract parameter: {}".format(candidate))
        total_reward = 0.0
        # Evaluate current candidate by performing multiple rollouts
        for _ in range(num_rollouts):
            obs, _ = env_copy.reset()
            done = False
            ep_reward = 0.0
            # Handle different observation formats
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
                    # Accumulate reward for this rollout
                    if isinstance(reward, dict):
                        ep_reward += sum(reward.values())
                    else:
                        ep_reward += reward
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
            best_contract_param = candidate

    logger.log_stat("solver_optimal_contract", best_contract_param, 0)
    # Finally, update the main environment's contract parameter with the best candidate.
    env_copy.update_contract(best_contract_param)
    return best_contract_param
