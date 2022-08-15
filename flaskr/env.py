import gym
import numpy as np
import torch
from pandas import isna

from gym import spaces
from influxdb_client import InfluxDBClient
from collections import deque

import time
from math import floor

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from .a2c_ppo_acktr.arguments import get_args
args = get_args()

class InfluxData():
    def __init__(self):
        self.space_keys = ['active_core_count', 'bytes_recv', 'bytes_sent', 'concurrency', 'dropin', 'dropout','jobId',
                           'rtt', 'latency', 'parallelism', 'pipelining', 'jobSize', 'packets_sent', 'packets_recv',
                           'errin', 'errout', 'totalBytesSent', 'memory', 'throughput', 'avgJobSize', 'freeMemory']

        self.client = InfluxDBClient.from_config_file("config.ini")
        self.query_api = self.client.query_api()

        self.p = {
            '_APP_NAME': "onedatashare@gmail.com-didclab-pred",
            '_TIME': '-2m',
        }

    def query_space(self):
        q='''from(bucket: "onedatashare@gmail.com")
  |> range(start: -5m)
  |> fill(usePrevious: true)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") 
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == _APP_NAME)'''

        data_frame = self.query_api.query_data_frame(q, params=self.p)
        # print(data_frame.tail())
        # print(data_frame.columns)
        # print(data_frame)
        return data_frame

    def prune_df(self, df):
        df2 = df[self.space_keys]
        # print(df2.tail())
        return df2

    def close_client(self):
        self.client.close()

class InfluxEnvironment(gym.Env):
    def __init__(self, max_cc, max_p, max_pp, influx_client, agent_train_callback, device):

        self.influx_client = influx_client
        self.train_callback = agent_train_callback

        self.max_concurrency = max_cc
        self.max_parallelism = max_p
        self.max_pipeline = max_pp
        self._parameter_ceiling = min(max_cc, max_p)

        self.possible_parameters = [i * 1. for i in range(1, 31)]
        self._parameter_baseline = 10.
        # CC, P for now
        self.action_space = spaces.Discrete(n=3)

        self._p_names = ['concurrency', 'parallelism', 'pipelining', 'chunkSize']
        self._running_obs = ['freeMemory', 'rtt', 'jobSize', 'avgJobSize', 'throughput']
        self._obs_names = ['concurrency', 'pipelining', 'parallelism'] + self._running_obs
        # CC, P, PP, totalBytesSent, memory
        self.observation_space = spaces.Box(low=-4., high=4., dtype=np.float32, shape=(len(self._obs_names),))

        self._throughput_baseline = 1e8 # 1e6
        self.throughput_list = deque([0. for _ in range(5)], maxlen=5)

        self.obs_norm_list = {name: deque([0.], maxlen=100) for name in self._obs_names}

        self.key_names = ['jobId', 'bytes_sent']
        self._data_keys = {n: 0 for n in self.key_names}
        self._done_ptr = 1
        self._done_switch = False

        self.current_action = args.starting_action
        self._prev_throughput = 0.

        self._device = device
        self._cur_reward = 0.
        self._eps_reward = 0.

        self.bootstrapping = True
        self.output = None

        self._recovery_reward = 3.
        self.rewards = deque([0.], maxlen=100)

        self.reg = 0.
        self.episode_count = 0

    def close(self):
        self.influx_client.close_client()
        writer.flush()
        writer.close()

    def parse_action(self, net_output):
        # return (torch.div(net_output, 6, rounding_mode='floor'), net_output % 6)
        return (net_output // 6, net_output % 6)

    def encode_actions(self, row):
        try:
            # action =  (self.possible_parameters.index(row['parallelism']) * 6) + \
            #     self.possible_parameters.index(row['concurrency'])
            action = self.possible_parameters.index(row['parallelism'])
        except:
            action = -1
        return action

    def normalize(self, value, history):
        return (value - np.mean(history)) / np.std(history)

    def reset(self):
        vect = [
            self.normalize(args.starting_action['concurrency'], self.possible_parameters),
            self.normalize(args.starting_action['pipelining'], self.possible_parameters),
            self.normalize(args.starting_action['parallelism'], self.possible_parameters),
            0.,
            0.,
            0.,
            0.,
            0.
        ]

        self.current_action = args.starting_action.copy()

        self._prev_throughput = 0.

        return torch.tensor(vect, device=self._device).unsqueeze(0)

    def suggest_parameters(self):
        print(self.current_action)
        return self.current_action
    
    def interpret(self, action):
        # cc_index, p_index = self.parse_action(action)

        # self.current_action['concurrency'] = 2. # self.possible_parameters[cc_index]
        cur_p = self.current_action['parallelism']
        num_choices = len(self.possible_parameters)
        if action == 1 and cur_p > 1.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['parallelism'] = self.possible_parameters[p_index - 1]
        elif action == 2 and cur_p < 30.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['parallelism'] = self.possible_parameters[min(p_index + 2, num_choices - 1)]
        # self.current_action['parallelism'] = self.possible_parameters[p_index]

        return self.current_action
    
    def input_step(self, input_req):
        # if self.bootstrapping:
        #     print('Bootstrapping...')
        #     self.bootstrapping = False
        #     return
        
        data = self.influx_client.prune_df(self.influx_client.query_space())
    
        try:
            reward_scalar = (input_req.throughput / self._throughput_baseline)
            self._recovery_reward = reward_scalar
        except:
            print('!! NAN Reward encountered; Dumping !!')
            print('!! Recovering !!')
            reward_scalar = self._recovery_reward
        
        reward_scalar = floor(reward_scalar)
            
        self._cur_reward += reward_scalar
        print('Intra-step Reward:', self._cur_reward)
        
        filtered_data = data[(data['jobId'] > self._data_keys['jobId'])].copy()
        # filtered_data = data.copy()
        encoded_action = -1 if filtered_data.empty else self.encode_actions(filtered_data.iloc[-1])

        if not filtered_data.empty:
            self.rewards.append(self._cur_reward)
            # self._cur_reward = self.normalize(self._cur_reward, self.rewards)
            self._eps_reward += self._cur_reward

            print('New Job, Step Reward:', self._cur_reward, 'Eps Reward:', self._eps_reward)

            # Register new keys
            self._data_keys['jobId'] = filtered_data['jobId'].iat[-1]
            self._data_keys['bytes_sent'] = filtered_data['bytes_sent'].iat[-1]
            # Normalize data
            filtered_data['concurrency'] = self.normalize(input_req.concurrency, self.possible_parameters)
            # filtered_data['parallelism'] = self.normalize(filtered_data['parallelism'], self.possible_parameters)
            filtered_data['pipelining'] = self.normalize(input_req.pipelining, self.possible_parameters)

            for o in self._running_obs:
                self.obs_norm_list[o] += list(filtered_data[o] / self._throughput_baseline)
                filtered_data[o] = self.normalize(filtered_data[o], self.obs_norm_list[o])
            
            # Calculate rewards
            reward = torch.tensor(self._cur_reward)

            next_observation = torch.as_tensor(filtered_data[self._obs_names].iloc[-1]).unsqueeze(0)

            done = True if self._done_ptr % 3 == 0 else False

            info = {} if not done else {
                'episode': {
                    'r': self._eps_reward
                }
            }

            # Call agent
            self.output = self.train_callback(next_observation, reward, done, info, encoded_action)

            self._cur_reward = 0.
            if done:
                self._eps_reward = 0.

            if self.output is not None:
                print('Agent chose:', self.output)
                self.interpret(self.output.item())
                self._done_ptr += 1

            print('Setting action to: ', self.current_action)
    
    def fetch_and_train(self):
        # if self.bootstrapping:
        #     print('Bootstrapping...')
        #     self.bootstrapping = False
        #     return
        
        data = self.influx_client.prune_df(self.influx_client.query_space())
        
        filtered_data = data[(data['jobId'] > self._data_keys['jobId']) |
            ((data['jobId'] == self._data_keys['jobId']) & (data['bytes_sent'] > self._data_keys['bytes_sent']) & 
                (data['parallelism'] > 0))].copy()
        # filtered_data = data[(data['jobId'] > self._data_keys['jobId'])].copy()
        # filtered_data = data.copy()
        encoded_action = -1 if filtered_data.empty else self.encode_actions(filtered_data.iloc[-1])

        if not filtered_data.empty and (not self.output or (filtered_data['parallelism'].iat[-1] == self.current_action['parallelism'])):
            print('Next Event Available for action', self.output)

            # Register new keys
            self._data_keys['jobId'] = filtered_data['jobId'].iat[-1]
            self._data_keys['bytes_sent'] = filtered_data['bytes_sent'].iat[-1]
            # Normalize data
            reward_scaled = (filtered_data['throughput'] / self._throughput_baseline).iat[-1]

            if isna(reward_scaled):
                print('!! NAN Reward encountered; Dumping !!')
                print('reward_scalar', reward_scaled)
                print('Last Row', filtered_data[self._obs_names].iloc[-1])
                print('!! Recovering !!')
                # reward_scalar = self.normalize(self._recovery_reward, self.rewards)
                reward_scaled = self._recovery_reward
                filtered_data = filtered_data.assign(throughput=3e8)
            
            self.throughput_list.append(reward_scaled)
            if not self._done_switch and self.output != None and self.output.item() == 1 and self.current_action['parallelism'] == 1.:
                reward_scalar = -1
            # elif reward_scaled >= np.mean(self.throughput_list):
            #     reward_scalar = 1.
            # else:
            #     reward_scalar = 0.
            else:
                reward_scalar = np.round(np.mean(self.throughput_list), 1)
            
            reward_scalar += (self.reg * filtered_data['parallelism'].iat[-1])
            print('Reward:', reward_scalar)
            
            self._prev_throughput = reward_scaled
            # self.throughput_list.append(self._prev_throughput)

            filtered_data['concurrency'] = self.normalize(filtered_data['concurrency'], self.possible_parameters)
            filtered_data['parallelism'] = self.normalize(filtered_data['parallelism'], self.possible_parameters)
            filtered_data['pipelining'] = self.normalize(filtered_data['pipelining'], self.possible_parameters)

            for o in self._running_obs:
                if o in ['jobSize', 'avgJobSize']:
                    filtered_data[o] /= 1e9
                elif o == 'throughput':
                    filtered_data[o] /= self._throughput_baseline
                else:
                    self.obs_norm_list[o] += list(filtered_data[o])
                    filtered_data[o] = self.normalize(filtered_data[o], self.obs_norm_list[o])
            
            # Calculate rewards
            
            reward = torch.tensor(reward_scalar)

            next_observation = torch.as_tensor(filtered_data[self._obs_names].iloc[-1]).unsqueeze(0)
            
            # num_rows = filtered_data.shape[0]
            # ceiling = self._done_ptr + num_rows
            # dones = [True if i % 3 == 0 else False for i in range(self._done_ptr, ceiling)]
            # self._done_ptr = ceiling

            done = True if self._done_switch else False
            self._done_switch = False

            info = {} if not done else {
                'episode': {
                    'r': self._eps_reward
                }
            }

            if done:
                writer.add_scalar("Train/episode_reward", self._eps_reward, self.episode_count)
                self.episode_count += 1
                next_observation = self.reset()

            # Call agent
            self.output = self.train_callback(next_observation, reward, done, info, encoded_action)
            if done:
                self._eps_reward = 0.

            if self.output is not None:
                print('Agent chose:', self.output)
                self.interpret(self.output.item())
                self._done_ptr += 1
                self._eps_reward += reward_scalar
            print('Setting action to: ', self.current_action)

# if __name__ == "__main__":
#     def cb(obs, reward, done, info):
#         print([obs, reward, done, info])
#         return 2
#     # env = TransferEnvironment(32, 32 ,32, cb)
#     # print(env.suggest((4, 4), 9854217.))
#     # print(env.suggest((8, 16), 1040602.))
# #     # print(env.suggest((8, 16), 10000000.))
#     client = InfluxData()
#     client.prune_df(client.query_space())
#     client.close_client()
#     env = InfluxEnvironment(32, 32, 32, client, cb, torch.device("cuda:0"))
#     env.fetch_and_train()
#     print(env.suggest_parameters())
    
#     print('Sleeping...')
#     time.sleep(30.)
#     print('Awake')
#     env.fetch_and_train()
#     print(env.suggest_parameters())

#     print('Sleeping...')
#     time.sleep(30.)
#     print('Awake')
#     env.fetch_and_train()
#     print(env.suggest_parameters())

#     print('Sleeping...')
#     time.sleep(30.)
#     print('Awake')
#     env.fetch_and_train()
#     print(env.suggest_parameters())
#     # data = client.prune_df(client.query_space())
#     # print(len(data['memory']))
    
#     # filter = data[data['memory'] > 83532600].copy()
#     # print(filter)
#     # print(filter.empty)
#     # filter['memory'] = (filter['memory'] - 83532600) / np.std(filter['memory'])
#     # print(filter.shape)
#     # for i in range(10):
#     #     print(filter.iloc[i])
#     # print(filter['memory'].iat[-1])
    
