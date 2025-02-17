import os
import copy
from epymarl.src.run import run
from src.contract.moca_solver import run_solver
from src.handles.handles import AbstractExperimentHandle


class MOCAExperimentHandle(AbstractExperimentHandle):
    def __init__(self, config_dict_list):
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

        exp_results = run(_run, exp_params, _log)

        # 仅在非 MOCA 模式下，在训练结束后调用求解器（原有逻辑）
        if (not exp_params.get("moca", False)) and exp_params.get("solver", False) and not exp_params.get("separate", False):
            solver_params = copy.deepcopy(exp_params)
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
            self.config_dict_list[index]["directory_name"] = self.config_dict_list[index]["parent_name"] + "/" + new_name
        else:
            self.config_dict_list[index]["directory_name"] = new_name
        return new_name
