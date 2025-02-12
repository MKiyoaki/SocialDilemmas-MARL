"""
experiment_handle.py

This module defines the AbstractExperimentHandle class, an abstract base class for experiment handlers.
It provides the basic interface for experiment scheduling and management.
"""

import json

class AbstractExperimentHandle:
    def __init__(self, config_path):
        """
        Initialize the experiment handle by loading the configuration dictionary list from the given file.

        Parameters:
            config_path (str): Path to the JSON configuration file.
        """
        with open(config_path, 'r') as f:
            self.config_dict_list = json.load(f)
        self.experiment_name = self.config_dict_list[0].get('experiment_name', 'default_experiment')

    def run_exp(self):
        """
        Run the main experiment.
        Must be implemented in the subclass.
        """
        raise NotImplementedError("run_exp() must be implemented by the subclass.")

    def hook_at_start(self):
        """
        Execute initialization actions at the start of the experiment.
        Must be implemented in the subclass.
        """
        raise NotImplementedError("hook_at_start() must be implemented by the subclass.")

    def hook_at_end(self):
        """
        Execute cleanup actions at the end of the experiment.
        Must be implemented in the subclass.
        """
        raise NotImplementedError("hook_at_end() must be implemented by the subclass.")

    def argument_parsing(self, config_dict):
        """
        Parse experiment arguments from the given configuration dictionary.
        Must be implemented in the subclass.

        Parameters:
            config_dict (dict): A dictionary of configuration parameters.

        Returns:
            Parsed parameters as a dictionary.
        """
        raise NotImplementedError("argument_parsing() must be implemented by the subclass.")
