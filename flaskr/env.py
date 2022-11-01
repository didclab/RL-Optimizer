from abc import ABC

import gym
import numpy as np
import torch
from pandas import isna, read_csv

from gym import spaces
from influxdb_client import InfluxDBClient
from collections import deque
from .a2c_ppo_acktr.arguments import get_args

from .sprout_constants import *
from .poisson import PoissonDistribution

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

args = get_args()


class ParameterDistributionMap:
    def __init__(self, override_max=None):
        self.PD_map = {}
        for p in range(1, MAX_PARALLELISM + 1):
            for c in range(1, MAX_CONCURRENCY + 1):
                self.PD_map[(p, c)] = PoissonDistribution(
                    MAX_THROUGHPUT if override_max is None else override_max,
                    NUM_DISCRETE,
                    NUM_WIENER_STEPS
                )
        self.recommendation = (2, 2)
        self.total_updates = 0

    def prune(self):
        if self.total_updates < 480:
            return

        popped_keys = []

        for p, c in self.PD_map:
            dist = self.PD_map[(p, c)]
            dist_updates = dist.num_updates

            if dist_updates / self.total_updates < 0.01:
                popped_keys.append((p, c))
            dist.num_updates = 0

        for key in popped_keys:
            self.PD_map.pop(key, None)

        self.total_updates = 0

    def get_best_parameter(self):
        return self.recommendation

    def calculate_best_parameter(self):
        best_parameter = (0, 0)
        mean = 0

        for p, c in self.PD_map:
            cur_mean = self.PD_map[(p, c)].mean()
            if mean < cur_mean:
                mean = cur_mean
                best_parameter = (p, c)

        return best_parameter

    def update_parameter_dist(self, p, c, n_units, time=30):
        if p == 0 or c == 0:
            return
        distribution = self.PD_map.get((p, c))
        if distribution is None:
            distribution = PoissonDistribution(
                MAX_THROUGHPUT,
                NUM_DISCRETE,
                NUM_WIENER_STEPS
            )
            self.PD_map[(p, c)] = distribution

        distribution.update_distribution(time, n_units)

        self.recommendation = self.calculate_best_parameter()
        self.total_updates += 1


def simple_apply(x, dist_map):
    p = x['parallelism']
    c = x['concurrency']
    n_units = (x['throughput'] * 1e-9 / 32) * 30
    dist_map.update_parameter_dist(p, c, n_units)
    return x


class InfluxData:
    def __init__(self, file_name=None):
        self.space_keys = ['active_core_count', 'bytes_recv', 'bytes_sent', 'concurrency', 'dropin', 'dropout', 'jobId',
                           'rtt', 'latency', 'parallelism', 'pipelining', 'jobSize', 'packets_sent', 'packets_recv',
                           'errin', 'errout', 'totalBytesSent', 'memory', 'throughput', 'avgJobSize', 'freeMemory']

        self.client = InfluxDBClient.from_config_file("config.ini")
        self.query_api = self.client.query_api()

        self.p = {
            '_APP_NAME': "elvisdav@buffalo.edu-didclab-elvis-uc",
            '_TIME': '-2m',
        }

        self.input_file = file_name

    def query_space(self):
        q = '''from(bucket: "elvisdav@buffalo.edu")
  |> range(start: -2m)
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == _APP_NAME)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''

        data_frame = self.query_api.query_data_frame(q, params=self.p)
        # print(data_frame.tail())
        # print(data_frame.columns)
        # print(data_frame)
        return data_frame

    def prune_df(self, df):
        df2 = df[self.space_keys]
        # print(df2.tail())
        return df2

    def read_file(self):
        """
        Reads CSV Influx file in working directory
        :return: Pandas dataframe
        """
        data_frame = None
        if self.input_file is not None:
            data_frame = read_csv(self.input_file)
        return data_frame

    def close_client(self):
        self.client.close()


class InfluxEnvironment(gym.Env, ABC):
    def __init__(self, max_cc, max_p, max_pp, influx_client, agent_train_callback, device,
                 override_max=None):

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
        self._running_obs = ['freeMemory', 'rtt', 'jobSize', 'avgJobSize', 'totalBytesSent']
        self._obs_names = ['concurrency', 'pipelining', 'parallelism'] + self._running_obs
        # CC, P, PP, totalBytesSent, memory
        self.observation_space = spaces.Box(low=-4., high=4., dtype=np.float32, shape=(len(self._obs_names),))

        self._throughput_baseline = 1e8  # 1e6
        self.throughput_list = deque([0. for _ in range(5)], maxlen=5)

        self.obs_norm_list = {name: deque([0.], maxlen=100) for name in self._obs_names}

        self.key_names = ['jobId', 'bytes_sent']
        self._data_keys = {n: 0 for n in self.key_names}
        self._done_ptr = 1
        self._done_switch = False

        self.parameter_dist_map = ParameterDistributionMap(override_max=override_max)
        self.best_start = self.parameter_dist_map.get_best_parameter()

        self.current_action = args.starting_action.copy()
        self.current_action['parallelism'] = self.best_start[0]
        self.current_action['concurrency'] = self.best_start[1]
        self.past_action = self.current_action.copy()
        self._prev_throughput = 0.

        self._device = device
        self._cur_reward = 0.
        self._eps_reward = 0.

        self.bootstrapping = True
        self.output_p = self.output_c = None

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
        return net_output // 6, net_output % 6

    def encode_actions(self, row):
        try:
            # action =  (self.possible_parameters.index(row['parallelism']) * 6) + \
            #     self.possible_parameters.index(row['concurrency'])
            action = self.possible_parameters.index(row['concurrency'])
        except:
            action = -1
        return action

    def normalize(self, value, history):
        return (value - np.mean(history)) / np.std(history)

    def set_best_action(self, p, c):
        self.best_start = (p, c)

    def reset(self):
        self.current_action = args.starting_action.copy()
        self.current_action['parallelism'] = self.best_start[0]
        self.current_action['concurrency'] = self.best_start[1]
        # print(self.current_action, self.best_start)

        vect = [
            self.normalize(self.current_action['concurrency'], self.possible_parameters),
            self.normalize(self.current_action['pipelining'], self.possible_parameters),
            self.normalize(self.current_action['parallelism'], self.possible_parameters),
            0.,
            0.,
            0.,
            0.,
            0.
        ]

        self._prev_throughput = 0.

        return torch.tensor(vect, device=self._device).unsqueeze(0)

    def suggest_parameters(self):
        print(self.current_action)
        return self.current_action

    def interpret(self, action_p, action_c):
        # cc_index, p_index = self.parse_action(action)

        # self.current_action['concurrency'] = 2. # self.possible_parameters[cc_index]
        cur_p = self.current_action['concurrency']
        num_choices = len(self.possible_parameters)
        if action_c == 1 and cur_p > 1.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['concurrency'] = self.possible_parameters[p_index - 1]
        elif action_c == 2 and cur_p < 30.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['concurrency'] = self.possible_parameters[min(p_index + 1, num_choices - 1)]
        elif action_c == 3 and cur_p < 30.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['concurrency'] = self.possible_parameters[min(p_index + 6, num_choices - 1)]

        cur_p = self.current_action['parallelism']
        num_choices = len(self.possible_parameters)

        if action_p == 1 and cur_p > 1.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['parallelism'] = self.possible_parameters[p_index - 1]
        elif action_p == 2 and cur_p < 30.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['parallelism'] = self.possible_parameters[min(p_index + 1, num_choices - 1)]
        elif action_p == 3 and cur_p < 30.:
            p_index = self.possible_parameters.index(cur_p)
            self.current_action['parallelism'] = self.possible_parameters[min(p_index + 6, num_choices - 1)]

        return self.current_action

    def fetch_and_train(self):
        # if self.bootstrapping:
        #     print('Bootstrapping...')
        #     self.bootstrapping = False
        #     return

        data = self.influx_client.prune_df(self.influx_client.query_space())

        filtered_data = data[(data['jobId'] > self._data_keys['jobId']) |
                             ((data['jobId'] == self._data_keys['jobId']) & (
                                     data['bytes_sent'] > self._data_keys['bytes_sent']) &
                              (data['concurrency'] > 0))].copy()
        # filtered_data = data[(data['jobId'] > self._data_keys['jobId'])].copy()
        # filtered_data = data.copy()
        encoded_action = -1 if filtered_data.empty else self.encode_actions(filtered_data.iloc[-1])

        if not filtered_data.empty and (
                not self.output_c or (filtered_data['concurrency'].iat[-1] == self.current_action['concurrency'])):
            print('Next Event Available for action', self.output_p, self.output_c)

            # Update Posterior distribution live_data = filtered_data[(filtered_data['parallelism'] > 0.) & (
            # filtered_data['throughput'] > 0.)].filter( items=['parallelism', 'concurrency', 'throughput'] )
            # live_data.apply(lambda x: simple_apply(x, self.parameter_dist_map), axis=1) if not self._done_switch:
            # self.best_start = self.parameter_dist_map.get_best_parameter()

            # Register new keys
            self._data_keys['jobId'] = filtered_data['jobId'].iat[-1]
            self._data_keys['bytes_sent'] = filtered_data['bytes_sent'].iat[-1]
            # Normalize data
            reward_scaled = filtered_data['totalBytesSent'] / args.ping_interval
            reward_scaled = (reward_scaled / self._throughput_baseline).iat[-1]

            if isna(reward_scaled):
                print('!! NAN Reward encountered; Dumping !!')
                print('reward_scalar', reward_scaled)
                print('Last Row', filtered_data[self._obs_names].iloc[-1])
                print('!! Recovering !!')
                # reward_scalar = self.normalize(self._recovery_reward, self.rewards)
                reward_scaled = self._recovery_reward
                filtered_data = filtered_data.assign(throughput=2e9)

            self.throughput_list.append(reward_scaled)
            if not self._done_switch and self.output_c is not None and self.output_c.item() == 1 and self.past_action[
                'concurrency'] == 1.:
                reward_scalar_c = -1
            # elif reward_scaled >= np.mean(self.throughput_list):
            #     reward_scalar = 1.
            # else:
            #     reward_scalar = 0.
            else:
                reward_scalar_c = np.mean(self.throughput_list)
                # if filtered_data['concurrency'].iat[-1] > 1:
                #     reward_scalar_c *= (1 - (1 / filtered_data['concurrency'].iat[-1]))
                # else:
                #     reward_scalar_c *= 0.3
                reward_scalar_c = np.round(reward_scalar_c, 1)

            if not self._done_switch and self.output_p is not None and self.output_p.item() == 1 and self.past_action[
                'parallelism'] == 1.:
                reward_scalar_p = -1
            # elif reward_scaled >= np.mean(self.throughput_list):
            #     reward_scalar = 1.
            # else:
            #     reward_scalar = 0.
            else:
                reward_scalar_p = np.mean(self.throughput_list)
                # if filtered_data['parallelism'].iat[-1] > 1:
                #     reward_scalar_p *= (1 - (1 / filtered_data['parallelism'].iat[-1]))
                # else:
                #     reward_scalar_p *= 0.3
                reward_scalar_p = np.round(reward_scalar_p, 1)

            # reward_scalar_c += (self.reg * filtered_data['concurrency'].iat[-1])
            # reward_scalar_p += (self.reg * filtered_data['parallelism'].iat[-1])

            print('Reward:', reward_scalar_p, reward_scalar_c)

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
                elif o == 'totalBytesSent':
                    filtered_data[o] /= self._throughput_baseline
                    filtered_data[o] /= args.ping_interval
                else:
                    self.obs_norm_list[o] += list(filtered_data[o])
                    filtered_data[o] = self.normalize(filtered_data[o], self.obs_norm_list[o])

            # Calculate rewards

            reward_p = torch.tensor(reward_scalar_p)
            reward_c = torch.tensor(reward_scalar_c)

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
                # print('Environment Reset:', self.current_action)

            # Call agent
            self.output_p, self.output_c = self.train_callback(next_observation, (reward_p, reward_c), done, info,
                                                               encoded_action)
            if done:
                self._eps_reward = 0.

            if self.output_p is not None:
                print('Agent chose:', (self.output_p, self.output_c))
                self.past_action = self.current_action.copy()
                self.interpret(self.output_p.item(), self.output_c.item())
                self._done_ptr += 1
                self._eps_reward += np.mean((reward_scalar_p, reward_scalar_c))
            print('Setting action to: ', self.current_action)
