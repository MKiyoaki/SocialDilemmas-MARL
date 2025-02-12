def get_solver_config_from_params(params_dict, checkpoint_paths):
    """
    Create a solver-specific configuration and an environment instance for evaluation,
    based on the parameters provided in the configuration table.

    Parameters:
        params_dict (dict): Experiment parameter dictionary.
        checkpoint_paths (list): List of checkpoint paths from training.

    Returns:
        solver_config (dict): A modified configuration for solver evaluation.
        env_copy: An environment instance created using env_REGISTRY and env_args.
    """
    # Clone the configuration
    solver_config = params_dict.copy()
    # For solver evaluation, force single-worker and disable GPU usage
    solver_config["num_workers"] = 1
    solver_config["num_gpus"] = 0

    # Create a new environment instance using the environment registry
    from epymarl.src.envs import REGISTRY as env_REGISTRY
    env_name = solver_config.get("env", None)
    if env_name is None:
        raise ValueError("Environment name ('env') must be specified in the configuration.")
    env_args = solver_config.get("env_args", {})
    # Create environment instance;
    env_copy = env_REGISTRY[env_name](**env_args)

    return solver_config, env_copy
