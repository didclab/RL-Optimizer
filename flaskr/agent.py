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
        
        print('Actions:', encoded_action, self.action_clone)
        if self.action_clone.item() != encoded_action:
            print('!! SKIPPING !!')
            return None

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
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[self.cur_step], self.rollouts.recurrent_hidden_states[self.cur_step],
                        self.rollouts.masks[self.cur_step])
            # self.action_q.append((
            #     self.value, self.action, self.action_log_prob, self.recurrent_hidden_states
            # ))
        except:
            print("NAN ERROR", reward)
            self.actor_critic = Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy}
            )
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

###
# def main():
#     args = get_args()

#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True

#     log_dir = os.path.expanduser(args.log_dir)
#     eval_log_dir = log_dir + "_eval"
#     utils.cleanup_log_dir(log_dir)
#     utils.cleanup_log_dir(eval_log_dir)

#     torch.set_num_threads(1)
#     device = torch.device("cuda:0" if args.cuda else "cpu")

#     envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
#                          args.gamma, args.log_dir, device, False)

#     actor_critic = Policy(
#         envs.observation_space.shape,
#         envs.action_space,
#         base_kwargs={'recurrent': args.recurrent_policy})
#     actor_critic.to(device)

#     agent = algo.A2C_ACKTR(
#         actor_critic,
#         args.value_loss_coef,
#         args.entropy_coef,
#         lr=args.lr,
#         eps=args.eps,
#         alpha=args.alpha,
#         max_grad_norm=args.max_grad_norm)

#     rollouts = RolloutStorage(args.num_steps, args.num_processes,
#                               envs.observation_space.shape, envs.action_space,
#                               actor_critic.recurrent_hidden_state_size)

#     obs = envs.reset()
#     rollouts.obs[0].copy_(obs)
#     rollouts.to(device)

#     episode_rewards = deque(maxlen=10)

#     start = time.time()
#     num_updates = int(
#         args.num_env_steps) // args.num_steps // args.num_processes
#     for j in range(num_updates):

#         for step in range(args.num_steps):
#             # Sample actions
#             with torch.no_grad():
#                 print(rollouts.obs[step])
#                 print(rollouts.recurrent_hidden_states[step])
#                 print(rollouts.masks[step])
#                 print("DONE")
#                 value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
#                     rollouts.obs[step], rollouts.recurrent_hidden_states[step],
#                     rollouts.masks[step])

#             # Obser reward and next obs
#             obs, reward, done, infos = envs.step(action)

#             for info in infos:
#                 if 'episode' in info.keys():
#                     episode_rewards.append(info['episode']['r'])

#             # If done then clean the history of observations.
#             masks = torch.FloatTensor(
#                 [[0.0] if done_ else [1.0] for done_ in done])
#             bad_masks = torch.FloatTensor(
#                 [[0.0] if 'bad_transition' in info.keys() else [1.0]
#                  for info in infos])
#             rollouts.insert(obs, recurrent_hidden_states, action,
#                             action_log_prob, value, reward, masks, bad_masks)

#         with torch.no_grad():
#             next_value = actor_critic.get_value(
#                 rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
#                 rollouts.masks[-1]).detach()

#         rollouts.compute_returns(next_value, args.use_gae, args.gamma,
#                                  args.gae_lambda, args.use_proper_time_limits)

#         value_loss, action_loss, dist_entropy = agent.update(rollouts)

#         rollouts.after_update()

#         # save for every interval-th episode or for the last epoch
#         if (j % args.save_interval == 0
#                 or j == num_updates - 1) and args.save_dir != "":
#             save_path = os.path.join(args.save_dir, args.algo)
#             try:
#                 os.makedirs(save_path)
#             except OSError:
#                 pass

#             torch.save([
#                 actor_critic,
#                 getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
#             ], os.path.join(save_path, args.env_name + ".pt"))

#         if j % args.log_interval == 0 and len(episode_rewards) > 1:
#             total_num_steps = (j + 1) * args.num_processes * args.num_steps
#             end = time.time()
#             print(
#                 "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
#                 .format(j, total_num_steps,
#                         int(total_num_steps / (end - start)),
#                         len(episode_rewards), np.mean(episode_rewards),
#                         np.median(episode_rewards), np.min(episode_rewards),
#                         np.max(episode_rewards), dist_entropy, value_loss,
#                         action_loss))


# if __name__ == "__main__":
#     main()
###