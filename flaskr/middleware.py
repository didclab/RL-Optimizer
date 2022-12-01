import os

import torch

from flaskr import Optimizer, CreateOptimizerRequest, InputOptimizerRequest, DeleteOptimizerRequest
from .another_bo import BayesianOptimizerOld

class OptimizerMap(object):
    def __init__(self):
        self.optimizer_map = {}
        self.node_id_to_optimizer = {}
        self.vda2c = "VDA2C"
        self.bo = "BO"

    def get_optimizer(self, node_id):
        return self.optimizer_map[node_id]

    def create_optimizer(self, create_req: CreateOptimizerRequest, override_max=None):
        if create_req.node_id not in self.optimizer_map:
            if create_req.optimizerType == self.vda2c:
                self.optimizer_map[create_req.node_id] = Optimizer(create_req, override_max=override_max)
                self.node_id_to_optimizer[create_req.node_id] = self.vda2c
                return True
            elif create_req.optimizerType == self.bo:
                # self.optimizer_map[create_req.node_id] = (create_req.optimizerType, BayesianOptimizer(create_req))
                oldBo = BayesianOptimizerOld()
                oldBo.create_optimizer(create_req)
                self.optimizer_map[create_req.node_id] = oldBo
                self.node_id_to_optimizer[create_req.node_id] = self.bo
                return True
            # elif create_req.optimizerType == "SGD":
            #     self.optimizer_map[create_req.node_id] = (create_req.optimizerType,Optimizer(create_req, override_max=override_max))
            #     return True
            # elif create_req.optimizerType == "MADDPG":
            #     self.optimizer_map[create_req.node_id] = (create_req.optimizerType,Optimizer(create_req, override_max=override_max))
            #     return True
            else:
                return False
            # Initialize Optimizer
        else:
            print("Optimizer already exists for", create_req.node_id)
            return False

    def input_optimizer(self, input_req: InputOptimizerRequest):
        optimizer = self.optimizer_map[input_req.node_id]
        if self.node_id_to_optimizer[input_req.node_id] == self.vda2c:
            return optimizer.envs.suggest_parameters()
        elif self.node_id_to_optimizer[input_req.node_id] == self.bo:
            print("Putting to the BO optimizer")
            return optimizer.input_optimizer(input_req)
        return

    def delete_optimizer(self, delete_req: DeleteOptimizerRequest, args):
        opt = self.optimizer_map[delete_req.node_id]
        if self.node_id_to_optimizer[delete_req.node_id] == self.vda2c:
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                opt.actor_critic
            ], os.path.join(save_path, args.env_name + ".pt"))

        elif self.node_id_to_optimizer[delete_req.node_id] == self.bo:
            print("RM Bo optimizer")
            bo_opt = self.optimizer_map[delete_req.node_id]
            bo_opt.delete_optimizer(delete_req)
            bo_opt.close()
            self.optimizer_map.pop(delete_req.node_id)

        return delete_req.node_id

    def true_delete(self, delete_req: DeleteOptimizerRequest):
        self.optimizer_map[delete_req.node_id].envs.close()
        self.optimizer_map.pop(delete_req.node_id)
        return delete_req.node_id

    def clean_all(self):
        for key in self.optimizer_map:
            self.optimizer_map[key].envs.close()
