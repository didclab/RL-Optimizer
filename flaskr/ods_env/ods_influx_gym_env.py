import json

import gym
import requests
from gym import spaces
import os
import numpy as np
import matplotlib.pyplot as plt

from flaskr.ods_env.influx_query import InfluxData
from ods_influx_parallel_env import TransferApplicationParams

headers = {"Content-Type": "application/json"}


def raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-2m",
            render_mode=None, action_space_discrete=False):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    influx_client = InfluxData(bucket_name=bucket_name, transfer_node_name=transfer_node_name, file_name=None,
                               time_window=time_window)
    env = InfluxEnv(influx_client=influx_client, render_mode=render_mode, action_space_discrete=action_space_discrete)
    return env


class InfluxEnv(gym.Env):

    def __init__(self, influx_client=InfluxData(), reward_function=lambda rtt, thrpt: (rtt * thrpt),
                 action_space_discrete=False, render_mode=None, time_window="-7d", cc_max=64, pp_max=50, p_max=64):
        super(InfluxEnv, self).__init__()
        self.influx_client = influx_client
        self.space_df = self.influx_client.query_space(
            time_window)  # gets last 7 days worth of data. Gonna be a lil slow to create
        self.data_columns = self.space_df.columns.values
        self.reward_function = reward_function  # this function is then used to evaluate the obs space of the last entry
        if action_space_discrete:
            self.action_space = spaces.Discrete(3)  # drop stay increase
        else:
            self.action_space = spaces.Box(low=1, high=32, shape=(3,))

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(
            len(self.data_columns),))  # So this is probably not defined totally properly as I assume dimensions are 0-infinity
        self.past_rewards = []
        self.past_actions = []
        self.render_mode = render_mode

    """
    Stepping the env and a live transfer behind the scenes.
    Params: action = {"concurrency":1-64, "parallelism":1-64, "pipelining:1-100}
    Returns: observation: last non-na entry from influx which contains the action for that obs
             reward: the throughput * rtt of the slowest link(source or destination)
             done: the official status if this job is done. If it is done than we need to reset() which should launch a job?
             info: Currently None and not necessary I believe.
    """

    def step(self, action):
        if not action:
            return {}, {}, {}, {}
        self.send_application_params_tuple(action['concurrency'], action['parallelism'], action['pipelining'], 0)
        self.past_actions.append(action)
        newer_df = self.influx_client.query_space("-1m")  # last min is 2 points.
        self.space_df.append(newer_df)
        self.space_df.drop_duplicates(inplace=True)
        self.space_df.dropna(inplace=True)
        # Need to loop while we have not gotten the next observation of the agents.
        last_row = self.space_df.tail(n=1)
        observation = last_row
        print("Observation: \n", observation)
        # lambda cc,pp,p,ck, RTT, thrpt
        # Here we need to use monitoring API to know when the job is formally done vs when its running very slow
        if self.query_if_job_done(last_row['jobId']):
            # then we are done
            terminated = True
        else:
            terminated = False
        if last_row['write_throughput'].iloc[-1] < last_row['read_throughput'].iloc[-1]:
            thrpt = last_row['write_throughput'].iloc[-1]
            rtt = last_row['destination_rtt'].iloc[-1]
        else:
            thrpt = last_row['read_throughput'].iloc[-1]
            rtt = last_row['source_rtt'].iloc[-1]

        reward = self.reward_function(rtt,
                                      thrpt)  # this reward is of the last influx column which is mapped to that observation so this is the past time steps not the current actions rewards

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        return observation, reward, terminated, None, None

    """
    So right now this does not launch another job. It simply resets the observation space to the last jobs influx entries
    """

    def reset(self):
        print("Past Actions: ", self.past_actions)
        print("Past Rewards: ", self.past_rewards)
        self.past_actions.clear()
        self.past_rewards.clear()

    """
    So this will support 3 writing modes:
    none: there is no rendering done
    graph: writes graphs of the df information and past actions and rewards
    ansi: prints the df information to standard out.
    """

    def render(self, mode="None"):
        print("Viewing the graphs on influxdb is probably best but here is stuff from the last epoch")
        self.space_df.plot()
        self.render()

    """
    Closes in the influx client behind the scenes
    """

    def close(self):
        self.influx_client.close_client()
