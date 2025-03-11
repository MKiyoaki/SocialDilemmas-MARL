import os

# directories of the project
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

output_dir = os.path.join(proj_dir, 'results')
config_dir = os.path.join(proj_dir, 'src', 'config')
