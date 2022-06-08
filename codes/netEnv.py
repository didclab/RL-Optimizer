# @Author: jamil
# @Date:   2022-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2022-07-04T14:44:57-05:00
import gym
from gym import spaces
import numpy as np
from fileData import environmentGroups
import random

THRUPUT_PENALTY = -0.5 # hyperparameter
MAX_TIMESTEPS = 100
MAX_ACTIONS = 173

"""
NetEnvironment Class is the gym environment for each environmentGroup
input: environment_group:   environment group for a key
group key: 'FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'
"""
class NetEnvironment(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, environment_group,group_keys):
        self.environment_group = environment_group
        self.group_keys=group_keys
        self.actions = self.environment_group.return_global_action_list()
        self.group_index=random.randint(0, len(self.group_keys)-1)
        self.states = self.environment_group.return_state_list(self.group_keys[self.group_index])
        self.max_throughput = self.environment_group.group_maximum_throughput(self.group_keys[self.group_index])
        self.max_throughput_parameters=self.environment_group.return_group_max_throughput_parameters(self.group_keys[self.group_index])
        self.environment_group_identification=self.environment_group.return_group_identification(self.group_keys[self.group_index])
        self.current_observation = self.states[0]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_timesteps = MAX_TIMESTEPS
        self.time = 0
        self.b = THRUPUT_PENALTY
        self.prev_throughput = -1.
        self.current_observation = np.asarray(self.states[0])
        self.obs_shape=(8,)

    def reset(self):
        self.time = 0
        self.group_index=random.randint(0, len(self.group_keys)-1)
        self.prev_throughput = -1
        self.states = self.environment_group.return_state_list(self.group_keys[self.group_index])
        self.max_throughput = self.environment_group.group_maximum_throughput(self.group_keys[self.group_index])
        self.max_throughput_parameters=self.environment_group.return_group_max_throughput_parameters(self.group_keys[self.group_index])
        self.environment_group_identification=self.environment_group.return_group_identification(self.group_keys[self.group_index])
        self.current_observation = self.states[0]
        return np.asarray(self.current_observation)

    def step(self, action):
        action = self.actions[action]
        try:
            throughputs = self.environment_group.return_group_key_throughput(self.group_keys[self.group_index],action)
            cur_throughput = max(throughputs)
            reward = cur_throughput / self.max_throughput
            self.prev_throughput = cur_throughput
            # try:
            #     throughput_sample=random.sample(throughputs,int(len(throughputs))/2)
            #     cur_throughput = random.choice(throughput_sample)
            #     reward = cur_throughput / self.max_throughput
            #     self.prev_throughput = cur_throughput
            # except:
            #     cur_throughput = random.choice(throughputs)
            #     reward = cur_throughput / self.max_throughput
            #     self.prev_throughput = cur_throughput
        except:
            reward = self.b
        self.time += 1
        if self.max_timesteps <= self.time:
            done = True
        else:
            done = False

        info = {'time': self.time, 'max_time': self.max_timesteps}
        self.current_observation[-3:] = action
        return np.asarray(self.current_observation), reward, done, info

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_max_throughput(self):
        return self.max_throughput

    def get_time(self):
        return self.time

    def render(self):
        pass
