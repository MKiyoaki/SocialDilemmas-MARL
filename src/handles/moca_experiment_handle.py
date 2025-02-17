import os
import copy
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
        self.hook_at_start()
        i, config_dict = 0, self.config_dict_list[0]
        self.experiment_name = self.get_exp_name(i)
        exp_params = self.argument_parsing(config_dict)
        print("Main Experiment started for:", self.experiment_name)

        # Run the online training phase (stage 1) using the MOCA learner
        exp_results = run(_run, exp_params, _log)

        # For MOCA, after training, call the solver (stage 2) to optimize the contract
        # Check if MOCA and solver flags are enabled and if not running in separate mode
        if exp_params.get("moca", False) and exp_params.get("solver", False) and not exp_params.get("separate", False):
            solver_params = copy.deepcopy(exp_params)
            # Pass candidate contracts from stage 1 if available
            if "candidate_contracts" in exp_results:
                solver_params["candidate_contracts"] = exp_results["candidate_contracts"]
            run_solver(solver_params, [exp_results["weight_directories"]], exp_results["logger"])

        print("Main Experiment ended for:", self.experiment_name)
        self.hook_at_end()
        return exp_results

    def argument_parsing(self, config_dict):
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
