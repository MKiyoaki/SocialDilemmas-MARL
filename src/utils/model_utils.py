"""
src/utils/model_utils.py
"""

import os
import torch as th
import numpy as np

from epymarl.src.controllers import REGISTRY as mac_REGISTRY
from epymarl.src.learners import REGISTRY as le_REGISTRY


class FrozenPolicy:
    def __init__(self, mac):
        self.mac = mac
        # Set the MAC's agent to evaluation mode
        self.mac.agent.eval()
        # Initialize hidden state with a batch_size of 1
        self.mac.init_hidden(1)

    def compute_action(self, obs):
        """
        Compute the action given an observation.
        obs: tuple, where the first element is a list of numpy arrays (one per agent),
             and the second element is additional info (unused).
        Returns: a numpy array of actions for each agent.
        """
        with th.no_grad():
            # Extract observation list. If obs[0] is not a list, wrap it in a list.
            if isinstance(obs[0], list):
                obs_list = obs[0]
            else:
                obs_list = [obs[0]]
            n_agents = len(obs_list)
            # If the number of observations does not match the expected number, replicate if necessary
            if n_agents != self.mac.args.n_agents:
                if n_agents == 1 and self.mac.args.n_agents > 1:
                    obs_list = [obs_list[0]] * self.mac.args.n_agents
                    n_agents = self.mac.args.n_agents
                else:
                    print("WARNING: Mismatch in number of agents. Proceeding with n_agents =", n_agents)

            # Convert each observation to a tensor and stack them to shape (n_agents, obs_dim)
            obs_stack = np.stack(obs_list)
            obs_tensor = th.tensor(obs_stack, dtype=th.float32, device=self.mac.args.device)
            # Add batch dimension, resulting in shape (1, n_agents, obs_dim)
            obs_tensor = obs_tensor.unsqueeze(0)

            batch_size = 1

            # If configuration requires last action observation, create a zeros tensor for actions_onehot
            if getattr(self.mac.args, "obs_last_action", False):
                actions_onehot = th.zeros((batch_size, n_agents, self.mac.args.n_actions),
                                          dtype=th.float32, device=self.mac.args.device)
            else:
                actions_onehot = None

            # Assume all actions are available; create avail_actions as a tensor of ones with shape (1, n_agents, n_actions)
            avail_actions = th.ones((batch_size, n_agents, self.mac.args.n_actions),
                                    dtype=th.float32, device=self.mac.args.device)

            # If agent ID information is required, create one-hot encoding with shape (1, n_agents, n_agents)
            if getattr(self.mac.args, "obs_agent_id", False):
                agent_ids = th.eye(n_agents, dtype=th.float32, device=self.mac.args.device).unsqueeze(0)
            else:
                agent_ids = None

            # Construct input list in the same order as BasicMAC._build_inputs
            inputs = [obs_tensor.view(batch_size * n_agents, -1)]
            if actions_onehot is not None:
                inputs.append(actions_onehot.view(batch_size * n_agents, -1))
            if agent_ids is not None:
                inputs.append(agent_ids.view(batch_size * n_agents, -1))

            # Concatenate to form input tensor of shape (batch_size*n_agents, input_dim)
            x = th.cat(inputs, dim=-1)

            # Initialize hidden state if it is not already initialized
            if self.mac.hidden_states is None:
                self.mac.init_hidden(batch_size)
            # Ensure hidden state is on the correct device
            self.mac.hidden_states = self.mac.hidden_states.to(self.mac.args.device)

            # Forward pass to compute logits
            agent_logits, self.mac.hidden_states = self.mac.agent(x, self.mac.hidden_states)

            # If output type is policy logits, apply masking and softmax
            if self.mac.agent_output_type == "pi_logits":
                # Reshape logits to (batch_size, n_agents, n_actions)
                agent_logits = agent_logits.view(batch_size, n_agents, -1)
                if getattr(self.mac.args, "mask_before_softmax", True):
                    avail_actions_reshaped = avail_actions.view(batch_size * n_agents, -1)
                    agent_logits_reshaped = agent_logits.view(batch_size * n_agents, -1)
                    # Set logits of unavailable actions to a very small value (assuming avail_actions == 0 indicates unavailable)
                    agent_logits_reshaped[avail_actions_reshaped == 0] = -1e10
                    agent_logits = agent_logits_reshaped.view(batch_size, n_agents, -1)
                # Compute probability distribution
                probs = th.softmax(agent_logits, dim=-1)
                # Select the action with the highest probability
                chosen_actions = th.argmax(probs, dim=-1)  # Shape: (1, n_agents)
            else:
                # If not logits, simply use argmax
                chosen_actions = th.argmax(agent_logits, dim=-1)
                chosen_actions = chosen_actions.view(batch_size, n_agents)

            # Return a numpy array with shape (n_agents,)
            return chosen_actions.squeeze(0).cpu().numpy()


def load_frozen_policy(solver_config, checkpoint_paths, env_info):
    """
    Load the frozen policy model from the provided checkpoint directory.
    The scheme is constructed based on env_info (obtained from runner.get_env_info() in run.py).
    """
    # For now, force using CPU; update for CUDA support later
    solver_config.device = "cpu"

    # Use the first checkpoint path if available, otherwise use the store_path from config
    checkpoint_dir = checkpoint_paths[0] if checkpoint_paths and len(checkpoint_paths) > 0 else solver_config.store_path

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if getattr(solver_config, "common_reward", False):
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (env_info["n_agents"],)}
    groups = {"agents": env_info["n_agents"]}
    # Create the multi-agent controller (MAC) using the same configuration as in run.py
    mac = mac_REGISTRY[solver_config.mac](scheme, groups, solver_config)
    learner_cls = le_REGISTRY["moca_learner"] if getattr(solver_config, "moca", False) else le_REGISTRY[solver_config.learner]
    learner = learner_cls(mac, scheme, None, solver_config)
    # Load the model parameters from the checkpoint directory (which should contain agent.th)
    learner.load_models(checkpoint_dir)
    return FrozenPolicy(learner.mac)
