import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import env
from .env import *

# from a2c_ppo_acktr import algo, utils
# from a2c_ppo_acktr.algo import gail
# from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs
# from a2c_ppo_acktr.model import Policy
# from a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate

# from classes import CreateOptimizerRequest
# from classes import DeleteOptimizerRequest
# from classes import InputOptimizerRequest

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate

from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest

optimizer_map = {}
args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

log_dir = os.path.expanduser(args.log_dir)
eval_log_dir = log_dir + "_eval"
utils.cleanup_log_dir(log_dir)
utils.cleanup_log_dir(eval_log_dir)

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

class Optimizer(object):
    def train(self, next_obs, reward, done, info, encoded_action):
        
        # print('Actions:', encoded_action, self.action_clone)
        # if self.action_clone.item() != encoded_action:
        #     print('!! SKIPPING !!')
        #     return None

        masks = torch.FloatTensor(
            [[0.0] if done else [1.0]])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]])

        # entry = self.action_q[1]
        self.rollouts.insert(next_obs, self.recurrent_hidden_states, self.action,
                            self.action_log_prob, self.value, reward, masks, bad_masks)

        if 'episode' in info.keys():
            self.episode_rewards.append(info['episode']['r'])
        
        if self.cur_step == args.num_steps:
            print('~~~ Updating Policy ~~~~')
            self.cur_step = 0

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()
            
            self.rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()

            if (self.cur_update % args.save_interval == 0
                or self.cur_update == self.num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    self.actor_critic
                ], os.path.join(save_path, args.env_name + ".pt"))

            if self.cur_update % 1 == 0 and len(self.episode_rewards) > 1:
                total_num_steps = (self.cur_update + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(self.cur_update, total_num_steps,
                            int(total_num_steps / (end - self.start)),
                            len(self.episode_rewards), np.mean(self.episode_rewards),
                            np.median(self.episode_rewards), np.min(self.episode_rewards),
                            np.max(self.episode_rewards), dist_entropy, value_loss,
                            action_loss))
            
            self.cur_update += 1
        
        try:
            if done:
                obs = self.envs.reset()
                self.rollouts.obs[self.cur_step].copy_(obs)
                self.rollouts.masks[self.cur_step].copy_(torch.FloatTensor([[1.0]]))

            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[self.cur_step], self.rollouts.recurrent_hidden_states[self.cur_step],
                        self.rollouts.masks[self.cur_step])
            # self.action_q.append((
            #     self.value, self.action, self.action_log_prob, self.recurrent_hidden_states
            # ))
        except:
            print("NAN ERROR", reward, self.cur_step)
            print("@@@ ROLLOUT OBS @@@", self.rollouts.obs[self.cur_step])
            print("@@@ ROLLOUT REC_HIDDEN @@@", self.rollouts.recurrent_hidden_states[self.cur_step])
            print("@@@ ROLLOUT MASKS @@@", self.rollouts.recurrent_hidden_states[self.cur_step])
            print("@@@ Attemping Recovery... @@@")
            self.rollouts.obs[self.cur_step][0, -1] = 3.
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[self.cur_step], self.rollouts.recurrent_hidden_states[self.cur_step],
                        self.rollouts.masks[self.cur_step])
            print("### Recovered Action:", self.action, "###")
            # self.actor_critic = Policy(
            #     self.envs.observation_space.shape,
            #     self.envs.action_space,
            #     base_kwargs={'recurrent': args.recurrent_policy}
            # )
            # self.actor_critic.to(device)

            # self.agent = algo.A2C_ACKTR(
            #     self.actor_critic,
            #     args.value_loss_coef,
            #     args.entropy_coef,
            #     lr=args.lr,
            #     eps=args.eps,
            #     alpha=args.alpha,
            #     max_grad_norm=args.max_grad_norm
            # )
        self.cur_step += 1
        self.action_clone = self.action.clone().detach()
        return self.action_clone


    def __init__(self, create_req: CreateOptimizerRequest):
        self.envs = InfluxEnvironment(
            create_req.max_concurrency,
            create_req.max_parallelism,
            create_req.max_pipesize,
            InfluxData(),
            self.train,
            device
        )
        
        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy}
        )
        
        # save_path = os.path.join(args.save_dir, args.algo)
        # self.actor_critic = torch.load(os.path.join(save_path, args.env_name + ".pt"))[0]
        
        self.actor_critic.to(device)

        self.agent = algo.A2C_ACKTR(
            self.actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm
        )

        self.rollouts = RolloutStorage(args.num_steps, 1,
                                self.envs.observation_space.shape, self.envs.action_space,
                                self.actor_critic.recurrent_hidden_state_size)

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(device)

        self.episode_rewards = deque(maxlen=10)

        self.start = time.time()
        self.num_updates = int(
            args.num_env_steps) // args.num_steps // 1
        
        self.cur_update = 0
        self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                    self.rollouts.obs[0], self.rollouts.recurrent_hidden_states[0],
                    self.rollouts.masks[0])
        self.num_steps = args.num_steps
        self.cur_step = 1
        self.action_clone = self.action.clone().detach()
        
        self.action_q = deque(maxlen=2)
        self.action_q.append((
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states
        ))

        self.envs.interpret(self.action.item())
        

def get_optimizer(node_id):
    return optimizer_map[node_id]

def create_optimizer(create_req: CreateOptimizerRequest):
    if create_req.node_id not in optimizer_map:
        # Initialize Optimizer
        optimizer_map[create_req.node_id] = Optimizer(create_req)
        return True
    else:
        print("Optimizer already exists for", create_req.node_id)
        return False

def input_optimizer(input_req: InputOptimizerRequest):
    opt = optimizer_map[input_req.node_id]
    # opt.envs.input_step(input_req)
    return opt.envs.suggest_parameters()

def delete_optimizer(delete_req: DeleteOptimizerRequest):
    opt = optimizer_map[delete_req.node_id]
    save_path = os.path.join(args.save_dir, args.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save([
        opt.actor_critic
    ], os.path.join(save_path, args.env_name + ".pt"))
    return delete_req.node_id

def clean_all():
    for key in optimizer_map:
        optimizer_map[key].envs.close()
