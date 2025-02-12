"""
moca_experiment_handle.py

This module implements the MOCAExperimentHandle class, a concrete experiment handler
for running MOCA-based MARL experiments without Ray. It orchestrates the experiment in two stages:
1. Training stage: running the training routine (via run() from run.py) that uses the MOCA learner.
2. Solver stage: after training, invoking the MOCA solver to determine the optimal contract.

All experiment parameters are parsed from the configuration dictionaries.
"""

from src.handles.handles import AbstractExperimentHandle
import os
import copy
from epymarl.src.run import run        # Your training routine in run.py
from src.contract.moca_solver import run_solver  # Your MOCA solver module

class MOCAExperimentHandle(AbstractExperimentHandle):
    def __init__(self, config_dict_list):
        """
        Initialize the MOCAExperimentHandle with a list of configuration dictionaries.

        Parameters:
            config_dict_list (list): A list of configuration dictionaries.
        """
        self.config_dict_list = config_dict_list

    def hook_at_start(self):
        """
        Execute initialization actions at the start of the experiment.
        """
        print("Starting MOCA experiment...")

    def hook_at_end(self):
        """
        Execute cleanup actions at the end of the experiment.
        """
        print("MOCA experiment finished. Cleaning up...")

    def run_exp(self, _run, _log):
        """
        Main execution loop for running the MOCA experiment.
        This method parses the experiment parameters, runs the training stage,
        and if enabled, invokes the solver stage.
        """
        # Pre-experiment initialization
        self.hook_at_start()
        # Use the first configuration dictionary for naming and parameter parsing
        i, config_dict = 0, self.config_dict_list[0]
        self.experiment_name = self.get_exp_name(i)
        exp_params = self.argument_parsing(config_dict)
        print("Main Experiment started for:", self.experiment_name)

        # Run training stage using the existing run() function (which internally uses MOCA Learner if configured)
        exp_results = run(_run, exp_params, _log)

        # If solver is enabled, run the solver stage to determine the optimal contract.
        if config_dict.get("solver", False) and not config_dict.get("separate", False):
            solver_params = copy.deepcopy(exp_params)
            run_solver(solver_params, exp_results["weight_directories"], exp_results["logger"])

        print("Main Experiment ended for:", self.experiment_name)
        self.hook_at_end()
        return exp_results

    def argument_parsing(self, config_dict):
        """
        Parse experiment parameters.
        In this simple version, we return the configuration dictionary directly.

        Parameters:
            config_dict (dict): The configuration dictionary.

        Returns:
            dict: The experiment parameters.
        """
        return config_dict

    def get_exp_name(self, index):
        """
        Generate a unique experiment name based on the configuration.

        Parameters:
            index (int): Index of the configuration dictionary to use.

        Returns:
            str: A unique experiment name.
        """
        initial_name = self.config_dict_list[index].get("experiment_name", "experiment")
        new_name = initial_name
        i = 0
        while os.path.exists("gifs/" + new_name):
            new_name = initial_name + "_" + str(i)
            i += 1
        self.config_dict_list[index]["experiment_name"] = new_name
        if self.config_dict_list[index].get("parent_name"):
            self.config_dict_list[index]["directory_name"] = self.config_dict_list[index]["parent_name"] + "/" + new_name
        else:
            self.config_dict_list[index]["directory_name"] = new_name
        return new_name
