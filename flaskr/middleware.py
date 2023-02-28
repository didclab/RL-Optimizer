import os
import time
import torch
import flaskr.ods_env.ods_helper as oh

from flaskr import Optimizer, CreateOptimizerRequest, InputOptimizerRequest, DeleteOptimizerRequest
from .ods_env import ods_influx_parallel_env
from flaskr.another_bo import BayesianOptimizerOld
from flaskr.runner import Trainer


class OptimizerMap(object):
    def __init__(self):
        self.optimizer_map = {}
        self.node_id_to_optimizer = {}
        self.vda2c = "VDA2C"
        self.bo = "BO"
        self.maddpg = "MADDPG"
        self.ddpg="DDPG"

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
            elif create_req.optimizerType == self.maddpg:
                env = ods_influx_parallel_env.raw_env()  # for now b/c its me the defaults for the env should be fine
                env.reset()  # this reset needs to not launch a job as we are just creating the env
                self.node_id_to_optimizer[create_req.node_id] = self.maddpg
                self.optimizer_map[create_req.node_id] = ["optimizer_agent_here",env]  # the map should store the agent to the env as the value
                return True
            elif create_req.optimizerType == self.ddpg:
                result, meta = oh.query_if_job_running(create_req.job_id)
                if not result:
                    oh.submit_transfer_request(meta)
                    time.sleep(15)
                trainer = Trainer(create_opt_request=create_req)
                print("Created trainer")
                self.node_id_to_optimizer[create_req.node_id] = self.ddpg
                self.optimizer_map[create_req.node_id] = trainer

                trainer.train()
            else:
                return False
        else:
            if create_req.optimizerType == self.ddpg:
                trainer = self.optimizer_map[create_req.node_id]
                trainer.set_create_request(create_opt_req=create_req)
                if not trainer.training_flag:
                    trainer.train()
            print("Optimizer already exists for", create_req.node_id)
            return False

    def input_optimizer(self, input_req: InputOptimizerRequest):
        optimizer = self.optimizer_map[input_req.node_id]
        if self.node_id_to_optimizer[input_req.node_id] == self.vda2c:
            return optimizer.envs.suggest_parameters()
        elif self.node_id_to_optimizer[input_req.node_id] == self.bo:
            print("Putting to the BO optimizer")
            return optimizer.input_optimizer(input_req)
        elif self.node_id_to_optimizer[input_req.node_id] == self.maddpg:
            print("MADDPG getting next params for TS")
            agent_env_dict = self.optimizer_map[input_req.node_id]
            return agent_env_dict[1].agent_actions_cache.pop()
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

        elif self.node_id_to_optimizer[delete_req.node_id] == self.maddpg:
            print("RM MADDPG we are not deleting just resetting the env so it can keep training")
            env = self.optimizer_map[delete_req.node_id][1]
            env.reset()  # this reset should actually construct a new to run.

        elif self.node_id_to_optimizer[delete_req.node_id] == self.ddpg:
            env = self.optimizer_map[delete_req.node_id][1]
            env.reset(options={'launch_job':True})

        return delete_req.node_id

    def true_delete(self, delete_req: DeleteOptimizerRequest):
        self.optimizer_map[delete_req.node_id].envs.close()
        self.optimizer_map.pop(delete_req.node_id)
        return delete_req.node_id

    def clean_all(self):
        for key in self.optimizer_map:
            self.optimizer_map[key].envs.close()
