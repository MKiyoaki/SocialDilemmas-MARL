#!/bin/bash -l
#SBATCH --output=/users/%u/projects/results/coins_%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu:1
module load anaconda3
module load cuda
module load cudnn

source $(conda info --base)/etc/profile.d/conda.sh

conda env list
conda activate sdmarl_shimmy

export PYTHONPATH=/users/k21133239/projects/SocialDilemmas-MARL:$PYTHONPATH

python ../src/main.py --config=mappo_coins --env-config=meltingpot with env_args.time_limit=300 env_args.key="coins"