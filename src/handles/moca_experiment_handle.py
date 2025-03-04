import os
import copy
import numpy as np
from epymarl.src.run import run
from src.contract.moca_solver import run_solver
from src.handles.handles import AbstractExperimentHandle


class MOCAExperimentHandle(AbstractExperimentHandle):
    def __init__(self, config_dict_list, **kwargs):
        super().__init__(**kwargs)
        self.config_dict_list = config_dict_list

    def hook_at_start(self):
        print("Starting MOCA experiment...")

    def hook_at_end(self):
        print("MOCA experiment finished. Cleaning up...")

    def run_exp(self, _run, _log):
        """
        Run the MOCA experiment.
        This method sets up the experiment configuration to include the new contract settings,
        ensuring that the contract class (which is now a function) is passed to other modules (e.g., gym wrapper, learner).
        """
        self.hook_at_start()
        i, config_dict = 0, self.config_dict_list[0]
        self.experiment_name = self.get_exp_name(i)

        # Modify experiment parameters to include contract related settings.
        # For instance, set the flag for MOCA, the contract type, the contract function, and the contract parameters range.
        # These settings will be used by the learner and gym wrapper.
        exp_params = self.argument_parsing(config_dict)

        # Ensure the MOCA flag is set in the experiment configuration.
        # The learner and gym wrapper will check this flag to decide whether to use the contract function.
        exp_params["moca"] = exp_params.get("moca", True)
        # Pass along the contract type and transfer function name (which will be used to get the corresponding contract function)
        # These keys should be consistent with the ones used in the learner and gym wrapper.
        exp_params["contract_type"] = exp_params.get("contract_type", "general")
        exp_params["transfer_function"] = exp_params.get("transfer_function", "default_transfer_function")
        # Also pass the contract parameter range and initial contract parameter if provided.
        exp_params["contract_params_range"] = exp_params.get("contract_params_range", (0.0, 1.0))
        if "chosen_contract_params" not in exp_params:
            # If not provided, sample one randomly within the range
            low, high = exp_params["contract_params_range"]
            exp_params["chosen_contract_params"] = float(np.random.uniform(low, high))

        print("Main Experiment started for:", self.experiment_name)
        # Run the online training phase (stage 1) using the MOCA learner.
        exp_results = run(_run, exp_params, _log)

        print("Main Experiment ended for:", self.experiment_name)
        self.hook_at_end()
        return exp_results

    def argument_parsing(self, config_dict):
        """
        Parse the experiment arguments from the configuration dictionary.
        For MOCA experiments, ensure that the configuration includes keys for contract settings.
        """
        # Here we simply return the config_dict without modification,
        # but you can add additional parsing or default value assignments if needed.
        return config_dict

    def get_exp_name(self, index):
        initial_name = self.config_dict_list[index].get("experiment_name", "experiment")
        new_name = initial_name
        i = 0
        while os.path.exists("gifs/" + new_name):
            new_name = initial_name + "_" + str(i)
            i += 1
        self.config_dict_list[index]["experiment_name"] = new_name
        if self.config_dict_list[index].get("parent_name"):
            self.config_dict_list[index]["directory_name"] = self.config_dict_list[index][
                                                                 "parent_name"] + "/" + new_name
        else:
            self.config_dict_list[index]["directory_name"] = new_name
        return new_name
