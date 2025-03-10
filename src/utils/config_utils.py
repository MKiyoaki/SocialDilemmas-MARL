import os
import yaml

from src.utils.configuration import config_dir
from types import SimpleNamespace as SN


def get_solver_config_from_params(params_dict, checkpoint_paths):
    """
    Create a solver-specific configuration and an environment instance for evaluation,
    based on the parameters provided in the configuration table.

    Parameters:
        params_dict (dict): Experiment parameter dictionary.
        checkpoint_paths (list): List of checkpoint paths from training.

    Returns:
        solver_config (SimpleNamespace): A modified configuration for solver evaluation.
        env_copy: An environment instance created using env_REGISTRY and env_args.
    """
    # Get the defaults from default.yaml
    with open(os.path.join(config_dir, "default.yaml"), "r") as f:
        try:
            default_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Get default env_args from default_config
    default_env_args = default_config.get("env_args", {}).copy()
    # If top-level keys "common_reward" and "reward_scalarisation" exist but are not in env_args,
    # merge them into default_env_args.
    for key in ["common_reward", "reward_scalarisation"]:
        if key in default_config and key not in default_env_args:
            default_env_args[key] = default_config[key]

    # Clone the configuration
    solver_config = params_dict.copy()
    # For solver evaluation, force single-worker and disable GPU usage
    solver_config["num_workers"] = 1
    solver_config["num_gpus"] = 0

    # Merge default env_args with provided env_args
    provided_env_args = solver_config.get("env_args", {})
    merged_env_args = default_env_args.copy()
    merged_env_args.update(provided_env_args)
    solver_config["env_args"] = merged_env_args

    # Create a new environment instance using the environment registry
    from src.envs import REGISTRY as env_REGISTRY
    env_name = solver_config.get("env", None)
    if env_name is None:
        raise ValueError("Environment name ('env') must be specified in the configuration.")

    env_copy = env_REGISTRY[env_name](**merged_env_args)
    # Get env_info from the environment
    env_info = env_copy.get_env_info()
    # Update solver_config with env_info so that attribute access is possible (e.g., solver_config.n_agents)
    solver_config["n_agents"] = env_info["n_agents"]
    solver_config["n_actions"] = env_info["n_actions"]
    solver_config["state_shape"] = env_info["state_shape"]
    solver_config["obs_shape"] = env_info["obs_shape"]

    # Convert solver_config dict to a SimpleNamespace for attribute access
    solver_args = SN(**solver_config)
    return solver_args, env_copy
