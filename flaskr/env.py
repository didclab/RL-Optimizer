import gym
import numpy as np
import torch

from gym import spaces
from influxdb_client import InfluxDBClient

import time


class InfluxData():
    def __init__(self):
        self.space_keys = ['active_core_count', 'bytes_recv', 'bytes_sent', 'concurrency', 'dropin', 'dropout','jobId',
                           'rtt', 'latency', 'parallelism', 'pipelining', 'jobSize', 'packets_sent', 'packets_recv',
                           'errin', 'errout', 'totalBytesSent', 'memory', 'throughput']

        self.client = InfluxDBClient.from_config_file("config.ini")
        self.query_api = self.client.query_api()

        self.p = {
            '_APP_NAME': "onedatashare@gmail.com-didclab-pred",
            '_TIME': '-5m',
        }

    def query_space(self):
        q='''from(bucket: "onedatashare@gmail.com")
  |> range(start: -1h)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") 
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == _APP_NAME)'''

        data_frame = self.query_api.query_data_frame(q, params=self.p)
        # print(data_frame.columns)
        # print(data_frame)
        return data_frame

    def prune_df(self, df):
        df2 = df[self.space_keys]
        # print(df2.head())
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

        self.possible_parameters = [1., 2., 4., 8., 16., 31.]
        self._parameter_baseline = 10.
        # CC, P for now
        self.action_space = spaces.Discrete(n=36)

        self._p_names = ['concurrency', 'parallelism', 'pipelining', 'chunkSize']
        self._obs_names = ['concurrency', 'parallelism', 'pipelining', 'totalBytesSent', 'memory']
        # CC, P, PP, totalBytesSent, memory
        self.observation_space = spaces.Box(low=-4., high=4., dtype=np.float32, shape=(len(self._obs_names),))

        self._throughput_baseline = 1e8 # 1e6
        self.throughput_list = []

        self.obs_norm_list = {name: [0.] for name in self._obs_names}

        self.key_names = ['jobId', 'bytes_sent']
        self._data_keys = {n: 0 for n in self.key_names}
        self._done_ptr = 1

        self.current_action = {
            'chunkSize': 6000000.0, # set this to starting parameters
            'concurrency': 1.0, # set this to starting parameters
            'parallelism': 8.0, # set this to starting parameters
            'pipelining': 6.0, # set this to starting parameters
        }

        self._device = device
        self._eps_reward = 0.

        self.bootstrapping = True
        self.output = None

    def close(self):
        self.influx_client.close_client()

    def parse_action(self, net_output):
        return (torch.div(net_output, 6, rounding_mode='floor'), net_output % 6)

    def encode_actions(self, row):
        try:
            action =  (self.possible_parameters.index(row['parallelism']) * 6) + \
                self.possible_parameters.index(row['concurrency'])
        except:
            action = -1
        return action

    def normalize(self, value, history):
        return (value - np.mean(history)) / np.std(history)

    def reset(self):
        vect = [
            self.normalize(1, self.possible_parameters),
            self.normalize(1, self.possible_parameters),
            self.normalize(1, self.possible_parameters),
            0.,
            0.
        ]
        return torch.tensor(vect, device=self._device).unsqueeze(0)

    def suggest_parameters(self):
        print(self.current_action)
        return self.current_action
    
    def interpret(self, action):
        p_index, cc_index = self.parse_action(action)

        self.current_action['concurrency'] = self.possible_parameters[cc_index]
        self.current_action['parallelism'] = self.possible_parameters[p_index]

        return self.current_action
    
    def fetch_and_train(self):
        # if self.bootstrapping:
        #     print('Bootstrapping...')
        #     self.bootstrapping = False
        #     return
        
        data = self.influx_client.prune_df(self.influx_client.query_space())
        
        filtered_data = data[(data['jobId'] > self._data_keys['jobId']) |
            ((data['jobId'] == self._data_keys['jobId']) & (data['bytes_sent'] > self._data_keys['bytes_sent']) & 
                (data['parallelism'] > 0))].copy()
        encoded_action = -1 if filtered_data.empty else self.encode_actions(filtered_data.iloc[-1])

        if self.output and self.output.item() == encoded_action:
            print('Next Event Available!')

            # Register new keys
            self._data_keys['jobId'] = filtered_data['jobId'].iat[-1]
            self._data_keys['bytes_sent'] = filtered_data['bytes_sent'].iat[-1]
            # Normalize data
            filtered_data['concurrency'] = self.normalize(filtered_data['concurrency'], self.possible_parameters)
            filtered_data['parallelism'] = self.normalize(filtered_data['parallelism'], self.possible_parameters)
            filtered_data['pipelining'] = self.normalize(filtered_data['pipelining'], self.possible_parameters)

            self.obs_norm_list['totalBytesSent'] += list(filtered_data['totalBytesSent'])
            self.obs_norm_list['memory'] += list(filtered_data['memory'])

            filtered_data['totalBytesSent'] = self.normalize(filtered_data['totalBytesSent'], self.obs_norm_list['totalBytesSent'])
            filtered_data['memory'] = self.normalize(filtered_data['memory'], self.obs_norm_list['memory'])
            # Calculate rewards
            reward = (filtered_data['throughput'] / self._throughput_baseline).iat[-1]
            self._eps_reward += reward
            reward = torch.tensor(reward)

            next_observation = torch.as_tensor(filtered_data[self._obs_names].iloc[-1]).unsqueeze(0)
            
            # num_rows = filtered_data.shape[0]
            # ceiling = self._done_ptr + num_rows
            # dones = [True if i % 3 == 0 else False for i in range(self._done_ptr, ceiling)]
            # self._done_ptr = ceiling

            done = True if self._done_ptr % 3 == 0 else False
            self._done_ptr += 1

            info = {} if not done else {
                'episode': {
                    'r': self._eps_reward
                }
            }

            # Call agent
            self.output = self.train_callback(next_observation, reward, done, info, encoded_action)
            if done:
                self._eps_reward = 0.

            if self.output:
                self.interpret(self.output)
            print('Setting action to: ', self.current_action)
         

class TransferEnvironment(gym.Env):
    def __init__(self, max_cc, max_p, max_pp, agent_train_callback):
        # self.influx_client = influx_client

        self.max_concurrency = max_cc
        self.max_parallelism = max_p
        self.max_pipeline = max_pp
        self._parameter_ceiling = min(max_cc, max_p)

        # self.complete_df = self.influx_client.query_space()
        # self.pruned_df = self.influx_client.prune_df(self.complete_df)

        self.action_space = spaces.Discrete(n=36)
        # self._normalized_bounds = np.array([4., 4.])
        self.observation_space = spaces.Box(low=-4., high=4., dtype=np.float32, shape=(2,))

        self.agent_callback = agent_train_callback

        self._p_names = ['concurrency', 'parallelism', 'pipelining', 'chunkSize']

        self._throughput_baseline = 1e6
        self.throughput_list = []
        
        self.possible_parameters = [1., 2., 4., 8., 16., 31.]
        self._parameter_baseline = 10

        self.tunable_parameter_list = {}
        self.tunable_parameter_list[self._p_names[0]] = [self._parameter_baseline]
        self.tunable_parameter_list[self._p_names[1]] = [self._parameter_baseline]

    def parse_action(self, net_output):
        return (net_output // 6, net_output % 6)

    def normalize(self, value, history):
        return (value - np.mean(history)) / np.std(history)
    
    #action is a tuple of either +, ~, - times 2
    def step(self, action):
        p_index, cc_index = self.parse_action(action)

        return {
            'chunkSize': 0.0, # set this to starting parameters
            'concurrency': self.possible_parameters[cc_index],
            'parallelism': self.possible_parameters[p_index],
            'pipelining': 0.0, # set this to starting parameters
        }

    def suggest(self, ts_parameters, ts_throughput):
        # Reward Calculation
        self.throughput_list.append(ts_throughput)
        reward = ts_throughput // self._throughput_baseline
        
        # Callback to agent to complete step
        cc, p = ts_parameters

        self.tunable_parameter_list[self._p_names[0]].append(cc)
        self.tunable_parameter_list[self._p_names[1]].append(p)

        normalized_cc = self.normalize(cc, self.possible_parameters)
        normalized_p = self.normalize(p, self.possible_parameters)

        return self.agent_callback(np.array([normalized_cc, normalized_p]), reward, True, {})
    
    def reset(self):
        self.throughput_list = []
        self.tunable_parameter_list[self._p_names[0]] = [self._parameter_baseline]
        self.tunable_parameter_list[self._p_names[1]] = [self._parameter_baseline]
        # self.df = self.influx_client.query_space()
        # return self.df
        initial_cc = 2. # set this to start parameters
        initial_p = 2. # set this to start parameters
        
        normalized_cc = self.normalize(initial_cc, self.possible_parameters)
        normalized_p = self.normalize(initial_p, self.possible_parameters)
        return np.array([normalized_cc, normalized_p])



# if __name__ == "__main__":
#     def cb(obs, reward, done, info):
#         print([obs, reward, done, info])
#         return 2
#     # env = TransferEnvironment(32, 32 ,32, cb)
#     # print(env.suggest((4, 4), 9854217.))
#     # print(env.suggest((8, 16), 1040602.))
#     # print(env.suggest((8, 16), 10000000.))
#     client = InfluxData()
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
    
