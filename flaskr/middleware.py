import os

import torch

from flaskr import Optimizer, CreateOptimizerRequest, InputOptimizerRequest, DeleteOptimizerRequest
from .bo_optimizer import BayesianOptimizer

class OptimizerMap(object):
    def __init__(self):
        self.optimizer_map = {}

    def get_optimizer(self, node_id) -> Optimizer:
        return self.optimizer_map[node_id]

    def create_optimizer(self, create_req: CreateOptimizerRequest, override_max=None):
        if create_req.node_id not in self.optimizer_map:
            if create_req.optimizerType == "VDA2C":
                self.optimizer_map[create_req.node_id] = (create_req.optimizerType,Optimizer(create_req, override_max=override_max))
                return True
            elif create_req.optimizerType == "BO":
                self.optimizer_map[create_req.node_id] = (create_req.optimizerType, BayesianOptimizer(create_req))
                return True
            elif create_req.optimizerType == "SGD":
                self.optimizer_map[create_req.node_id] = (create_req.optimizerType,Optimizer(create_req, override_max=override_max))
                return True
            elif create_req.optimizerType == "MADDPG":
                self.optimizer_map[create_req.node_id] = (create_req.optimizerType,Optimizer(create_req, override_max=override_max))
                return True
            else:
                return False
            # Initialize Optimizer
        else:
            print("Optimizer already exists for", create_req.node_id)
            return False

    def input_optimizer(self, input_req: InputOptimizerRequest):
        optimizer_tuple = self.optimizer_map[input_req.node_id]
        print(optimizer_tuple)
        if optimizer_tuple[0] == "VDA2C":
            optimizer_tuple[1].envs.suggest_parameters()
        elif optimizer_tuple[0] == "BO":
            print("Inpuuting to BO optimizer")
            bo_opt = optimizer_tuple[1]
            bo_opt.input_optimizer(input_req)
        # elif optimizer_tuple[0] == "SGD":
        #     sgd_opt = optimizer_tuple[1]
        # elif optimizer_tuple[0] == "MADDPG":
        #     maddpg_opt = optimizer_tuple[1]
        return

    def delete_optimizer(self, delete_req: DeleteOptimizerRequest, args):
        optimizer_tuple = self.optimizer_map[delete_req.node_id]
        if optimizer_tuple[0] == "VDA2C":
            opt = optimizer_tuple[1]
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                opt.actor_critic
            ], os.path.join(save_path, args.env_name + ".pt"))

        elif optimizer_tuple[0] == "BO":
            print("Inpuuting to BO optimizer")
            bo_opt = optimizer_tuple[1]
            bo_opt.close()
            return delete_req.node_id

        elif optimizer_tuple[0] == "SGD":
            pass

        elif optimizer_tuple[0] == "MADDPG":
            pass

        return delete_req.node_id

    def true_delete(self, delete_req: DeleteOptimizerRequest):
        self.optimizer_map[delete_req.node_id].envs.close()
        self.optimizer_map.pop(delete_req.node_id)
        return delete_req.node_id

    def clean_all(self):
        for key in self.optimizer_map:
            self.optimizer_map[key].envs.close()
