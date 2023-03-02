import time

import numpy
import torch
import numpy as np
import os
from threading import Thread
from .ods_env import ods_influx_gym_env
from flaskr import classes
from .algos.ddpg import agents
from .algos.ddpg import memory
from .algos.ddpg import utils
from .ods_env import ods_helper as ods


class Trainer(object):
    def __init__(self, create_opt_request=classes.CreateOptimizerRequest, max_episodes=100, batch_size=64,
                 update_policy_time_steps=20):
        self.obs_cols = ['active_core_count', 'allocatedMemory',
                         'bytes_recv', 'bytes_sent', 'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min',
                         'dropin', 'dropout', 'errin', 'errout', 'freeMemory', 'maxMemory', 'memory',
                         'packet_loss_rate', 'packets_recv',
                         'packets_sent', 'bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency',
                         'destination_latency', 'destination_rtt',
                         'jobSize', 'parallelism', 'pipelining', 'read_throughput', 'source_latency', 'source_rtt',
                         'write_throughput']
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.create_opt_request = create_opt_request
        self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request, time_window="-1d",
                                                observation_columns=self.obs_cols)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = agents.DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=self.device, max_action=None)
        self.replay_buffer = memory.ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
        self.save_file_name = f"DDPG_{'influx_gym_env'}"
        try:
            os.mkdir('./models')
        except Exception as e:
            pass
        self.training_flag = False
        # self.worker_thread = Thread(target=self.train, args=(max_episodes, batch_size, update_policy_time_steps))

    def train(self, max_episodes=100, batch_size=64, update_policy_time_steps=2, launch_job=False):
        self.training_flag = True
        state = self.env.reset(options = {'launch_job': launch_job})  # no seed the first time
        print("State in train(): ", state, "\t", "type: ", type(state))
        state = np.asarray(a=state[0], dtype=numpy.float64)
        episode_reward = 0
        episode_num = 0
        episode_ts = 0
        episode_rewards = []
        # iterate until we hit max jobs
        while episode_num < max_episodes:
            if episode_ts < update_policy_time_steps:
                action = self.env.action_space.sample()
                print("train(): env selected action:", action)
            else:
                action = (self.agent.select_action(np.array(state)))
                print("train(): agent selected action:", action)

            prev_state, prev_reward, terminated, _ = self.env.step(action)
            print("Previous state: ", prev_state)
            print("Previous reward: ", prev_reward)
            print("Terminated: ", terminated)

            episode_ts += 1
            self.replay_buffer.add(state, action, prev_state, prev_reward, terminated)
            state = prev_state
            episode_reward += prev_reward
            if terminated:
                self.env.reset(options={'launch_job': True})
                print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_ts = 0

            if episode_ts % update_policy_time_steps == 0:
                self.agent.train(self.replay_buffer, batch_size)

        self.training_flag = False
        return episode_rewards

    def evaluate(self):
        avg_reward = utils.evaluate_policy(policy=self.agent, env=self.env)
        self.agent.save_checkpoint(f"./models/{self.save_file_name}")
        return avg_reward

    def set_create_request(self, create_opt_req):
        print("Updating the create_optimizer request in trainer and env to be: ", create_opt_req.__str__())
        self.create_opt_request = create_opt_req
        self.env.create_opt_request = create_opt_req
        self.env.job_id = create_opt_req.job_id
