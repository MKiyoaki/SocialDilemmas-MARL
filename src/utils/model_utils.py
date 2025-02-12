# src/utils/model_utils.py

import torch as th


class FrozenPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_action(self, obs):
        """
        Compute action given observation.
        This function converts the observation to a tensor (if needed),
        performs a forward pass through the model, and returns the action.
        """
        # Convert observation to tensor if it's not already one
        if not isinstance(obs, th.Tensor):
            obs_tensor = th.tensor(obs, dtype=th.float32)
        else:
            obs_tensor = obs
        with th.no_grad():
            action = self.model(obs_tensor)
        # Return action as a numpy array
        return action.cpu().numpy()


def load_frozen_policy(store_path, checkpoint_paths):
    """
    Load the frozen policy model from the provided store path or checkpoint paths.
    If checkpoint_paths is non-empty, load the first checkpoint; otherwise, use store_path.

    Parameters:
        store_path (str): The path specified in the configuration to store cache model checkpoints.
        checkpoint_paths (list): List of checkpoint file paths from training.

    Returns:
        FrozenPolicy: An instance of FrozenPolicy wrapping the loaded model.
    """
    if checkpoint_paths and len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[0]
    else:
        checkpoint_path = store_path  # Fallback to store_path if checkpoint_paths is empty

    model = th.load(checkpoint_path, map_location="cpu")
    return FrozenPolicy(model)
