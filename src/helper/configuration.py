import os

"""
Path Configuration
"""


# project dir
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_dir = os.path.join(proj_dir, '../')

# output dir
environment = "prisons_dilemmas_2" # "clean_up_2" #
environment_name = "prisoners_dilemma_in_the_matrix__repeated" # "clean_up"
algo_name = "PPO"

log_dir = os.path.join(proj_dir, f"output/logs/{environment}/{algo_name}")
model_dir = os.path.join(proj_dir, f"output/models/{environment}/{algo_name}")
