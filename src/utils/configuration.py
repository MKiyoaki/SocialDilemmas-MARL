import os

# directories of the project
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
epymarl_dir = os.path.join(proj_dir, 'epymarl')

config_dir = os.path.join(epymarl_dir, 'src', 'config')
