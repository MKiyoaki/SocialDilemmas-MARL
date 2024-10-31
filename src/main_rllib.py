import os

import torch
import ray

from ray import tune, air
from ray.rllib.algorithms import ppo
from ray.rllib.policy import policy

from harl.algorithms.actors import happo
from meltingpot.meltingpot import substrate

from src.helper import utils
from src.helper.configuration import proj_dir, environment_name, output_dir, output_name, algo_name

"""
Configuration
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_env_runner = 8
num_iterations = 1

rollout_fragment_length = 100
train_batch_size = 6400



def get_config(
    algorithm_config,
    substrate_name: str = environment_name,
    num_env_runners: int = num_env_runner,
    rollout_fragment_length: int = rollout_fragment_length,
    train_batch_size: int = train_batch_size,
    sgd_minibatch_size: int = 128,
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
):
    config = algorithm_config
    # Number of arenas.
    config.num_env_runners = num_env_runners
    # This is to match our unroll lengths.
    config.rollout_fragment_length = rollout_fragment_length
    # Total (time x batch) timesteps on the learning update.
    config.train_batch_size = train_batch_size
    # Mini-batch size.
    config.sgd_minibatch_size = sgd_minibatch_size
    # Use the raw observations/actions as defined by the environment.
    config.preprocessor_pref = None
    # Use TensorFlow as the tensor framework.
    config = config.framework("torch")
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config.num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "1"))

    # 2. Set environment config. This will be passed to
    # the env_creator function via the register env lambda below.
    player_roles = substrate.get_config(substrate_name).default_player_roles
    config.env_config = {"substrate": substrate_name, "roles": player_roles}

    config.env = "meltingpot"

    # 4. Extract space dimensions
    test_env = utils.env_creator(config.env_config)
    # print("Observation space for test_env:", test_env.observation_space)

    # Setup PPO with policies, one per entry in default player roles.
    policies = {}
    player_to_agent = {}

    for i in range(len(player_roles)):
        rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0] // 8
        sprite_y = rgb_shape[1] // 8

        policies[f"agent_{i}"] = policy.PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space[f"player_{i}"],
            action_space=test_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [[16, [8, 8], 8],
                                     [128, [sprite_x, sprite_y], 1]],
                },
            })
        player_to_agent[f"player_{i}"] = f"agent_{i}"

    def policy_mapping_fn(agent_id, episode, **kwargs):
        del episode
        del kwargs  # Ignore the useless params
        return player_to_agent[agent_id]

    # 5. Configuration for multi-agent setup with one policy per role:
    config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

    return config


def train(config, num_iterations):
    """Trains a model.

        Args:
            config: model config
            num_iterations: number of iterations ot train for.

        Returns:
            Training results.
    """
    tune.register_env("meltingpot", utils.env_creator)
    ray.init()
    stop = {
        "training_iteration": num_iterations,
    }
    return tune.Tuner(
        config.algo_name,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=0,
            storage_path=output_dir,
            name=output_name,
        ),
    ).fit()


def main():
    algorithm = ppo
    algo_config = algorithm.PPOConfig()
    algo_config.algo_name = algo_name

    config = get_config(algo_config)
    results = train(config, num_iterations)

    print(results)

    assert results.num_errors == 0


if __name__ == '__main__':
    main()
