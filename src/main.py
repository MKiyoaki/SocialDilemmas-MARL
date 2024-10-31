# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary to run Stable Baselines 3 agents on meltingpot substrates."""

from meltingpot.meltingpot import substrate
import stable_baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common import vec_env
import supersuit as ss
import torch

from src.helper import utils
from src.helper.cnn_structure import CustomCNN
from src.helper.configuration import environment_name, log_dir, model_dir

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
  # Config
  env_name = environment_name
  env_config = substrate.get_config(env_name)
  env = utils.parallel_env(env_config, env_name, render_mode="rgb_array")
  rollout_len = 1000
  total_timesteps = 2000000
  num_agents = env.max_num_agents

  # Training
  num_cpus = 1  # number of cpus
  num_envs = 1  # number of parallel multi-agent environments
  # number of frames to stack together; use >4 to avoid automatic
  # VecTransposeImage
  num_frames = 4
  # output layer of cnn extractor AND shared layer for policy and value
  # functions
  features_dim = 128
  fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
  ent_coef = 0.001  # entropy coefficient in loss
  batch_size = (rollout_len * num_envs // 2
               )  # This is from the rllib baseline implementation
  lr = 1e-4
  n_epochs = 30
  gae_lambda = 1.0
  gamma = 0.99
  target_kl = 0.01
  grad_clip = 40
  verbose = 1 # 3
  is_load = False

  model_path = model_dir  # Replace this with a saved model
  tensorboard_log = log_dir


  env = utils.parallel_env(
      max_cycles=rollout_len,
      env_config=env_config,
      env_name=env_name,
      render_mode="rgb_array"
  )
  env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
  env = ss.frame_stack_v1(env, num_frames)
  env = ss.pettingzoo_env_to_vec_env_v1(env)
  env = ss.concat_vec_envs_v1(
      env,
      num_vec_envs=num_envs,
      num_cpus=num_cpus,
      base_class="stable_baselines3",
  )
  env = vec_env.VecMonitor(env)
  env = vec_env.VecTransposeImage(env, True)

  eval_env = utils.parallel_env(
      max_cycles=rollout_len,
      env_config=env_config,
      env_name=env_name,
      render_mode="human"
  )
  eval_env = ss.observation_lambda_v0(eval_env, lambda x, _: x["RGB"],
                                      lambda s: s["RGB"])
  eval_env = ss.frame_stack_v1(eval_env, num_frames)
  eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
  eval_env = ss.concat_vec_envs_v1(
      eval_env,
      num_vec_envs=1,
      num_cpus=1,
      base_class="stable_baselines3"
  )
  eval_env = vec_env.VecMonitor(eval_env)
  eval_env = vec_env.VecTransposeImage(eval_env, True)
  eval_freq = 100000 // (num_envs * num_agents)

  policy_kwargs = dict(
      features_extractor_class=CustomCNN,
      features_extractor_kwargs=dict(
          features_dim=features_dim,
          num_frames=num_frames,
          fcnet_hiddens=fcnet_hiddens,
      ),
      net_arch=[features_dim],
  )

  model = stable_baselines3.PPO(
      "CnnPolicy",
      env=env,
      learning_rate=lr,
      n_steps=rollout_len,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      gae_lambda=gae_lambda,
      ent_coef=ent_coef,
      max_grad_norm=grad_clip,
      target_kl=target_kl,
      policy_kwargs=policy_kwargs,
      tensorboard_log=tensorboard_log,
      verbose=verbose,
  )
  if is_load is True and model_path is not None:
    model = stable_baselines3.PPO.load(model_path, env=env)
  eval_callback = callbacks.EvalCallback(
      eval_env, eval_freq=eval_freq, best_model_save_path=tensorboard_log)
  model.learn(total_timesteps=total_timesteps, callback=eval_callback)

  logdir = model.logger.dir
  model.save(logdir + "/model")
  del model
  stable_baselines3.PPO.load(logdir + "/model")


if __name__ == "__main__":
  main()
