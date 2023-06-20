import time

import gym
from gym import spaces
import numpy as np
import pandas as pd
from flaskr.classes import CreateOptimizerRequest
import flaskr.ods_env.ods_helper as oh
from flaskr.ods_env import env_utils
from flaskr.ods_env.influx_query import InfluxData

from .ods_rewards import DefaultReward, ArslanReward, JacobReward
import requests

requests.packages.urllib3.disable_warnings()

headers = {"Content-Type": "application/json"}

# display all the  rows
pd.set_option('display.max_rows', None)

# display all the  columns
pd.set_option('display.max_columns', None)

# set width  - 100
pd.set_option('display.width', 100)
# set column header -  left
pd.set_option('display.colheader_justify', 'left')

# set precision - 5
pd.set_option('display.precision', 5)


class InfluxEnv(gym.Env):

    def __init__(self, create_opt_req: CreateOptimizerRequest,
                 action_space_discrete=False, render_mode=None, time_window="-2m", observation_columns=[]):
        super(InfluxEnv, self).__init__()
        self.create_opt_request = create_opt_req
        print(create_opt_req.node_id.split("-"))
        bucket_name = create_opt_req.node_id.split("-")[0]
        self.influx_client = InfluxData(bucket_name=bucket_name, transfer_node_name=create_opt_req.node_id,
                                        file_name=None, time_window=time_window)
        # gets last 7 days worth of data. Gonna be a lil slow to create
        self.space_df = self.influx_client.query_space(time_window)
        self.job_id = create_opt_req.job_id
        if len(observation_columns) > 0:
            self.data_columns = observation_columns
        else:
            self.data_columns = self.space_df.columns.values
        if action_space_discrete:
            self.action_space = spaces.Discrete(3)  # drop stay increase
        else:
            self.action_space = spaces.Box(low=1, high=32, shape=(2,))  # for now cc, p only
        # So this is probably not defined totally properly as I assume dimensions are 0-infinity
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.data_columns),))
        self.past_rewards = []
        self.past_actions = []
        self.render_mode = render_mode

        self.drop_in = 0
        self.past_utility = 0
        print("Finished constructing the Influx Gym Env")

    """
    Stepping the env and a live transfer behind the scenes.
    Params: action = {"concurrency":1-64, "parallelism":1-64, "pipelining:1-100}
            reward_type = None | 'default' | 'arslan' (default: None)
    Returns: observation: last non-na entry from influx which contains the action for that obs
             reward: the throughput * rtt of the slowest link(source or destination)
             done: the official status if this job is done. If it is done than we need to reset() which should launch a job?
             info: Currently None and not necessary I believe.
    """

    def step(self, action, reward_type=None):
        print("Step: action:", action)
        self.space_df = self.influx_client.query_space("-60s")
        self.space_df.drop_duplicates(inplace=True)
        self.space_df.dropna(inplace=True)

        # Need to loop while we have not gotten the next observation of the agents.
        print(self.space_df[['read_throughput', "write_throughput"]])
        last_row = self.space_df.tail(n=1)
        observation = last_row[self.data_columns]
        diff_drop_in = last_row['dropin'].iloc[-1] - self.drop_in


        self.job_id = last_row['jobId']

        if action[0] < 1 or action[1] < 1 or action[0] > self.create_opt_request.max_concurrency or action[
            1] > self.create_opt_request.max_parallelism:
            reward = -10000000

        else:
            if int(last_row['concurrency']) != action[0] or int(last_row['parallelism']) != action[1]:
                oh.send_application_params_tuple(
                    transfer_node_name=self.create_opt_request.node_id,
                    cc=action[0], p=action[1], pp=1, chunkSize=0
                )

            if reward_type == 'arslan':
                reward_params = ArslanReward.Params(
                    penalty=diff_drop_in,
                    throughput=env_utils.smallest_throughput_rtt(last_row=last_row),
                    past_utility=self.past_utility,
                    concurrency=last_row['concurrency'].iloc[-1],
                    parallelism=last_row['parallelism'].iloc[-1]
                )

                reward, self.past_utility = ArslanReward.calculate(reward_params)
                self.drop_in += diff_drop_in
                self.past_actions.append(action)

            elif reward_type == 'default':
                thrpt, rtt = env_utils.smallest_throughput_rtt(last_row=last_row)
                self.past_actions.append(action)
                thrpt = thrpt / 8

                reward_params = DefaultReward.Params(
                    rtt=rtt,
                    throughput=thrpt
                )
                reward = DefaultReward.calculate(reward_params)

            elif reward_type == 'jacob':
                thrpt, rtt = env_utils.smallest_throughput_rtt(last_row=last_row)
                self.past_actions.append(action)
                reward_params = JacobReward.Params(
                    throughput=thrpt,
                    rtt=rtt,
                    c=last_row['concurrency'].iloc[-1],
                    max_cc=self.create_opt_request.max_concurrency,
                    p=last_row['parallelism'].iloc[-1],
                    max_p=self.create_opt_request.max_parallelism,
                    max_cpu_freq=last_row['cpu_frequency_max'].iloc[-1],
                    min_cpu_freq=last_row['cpu_frequency_min'].iloc[-1],
                    cpu_freq=last_row['cpu_frequency_current'].iloc[-1]
                )

                reward = JacobReward.calculate(reward_params)

        print("Step reward: ", reward)
        self.past_rewards.append(reward)

        while True:
            self.space_df = self.influx_client.query_space("-30s")
            self.space_df.drop_duplicates(inplace=True)
            self.space_df.dropna(inplace=True)
            cur_row = self.space_df.tail(n=1)
            observation = last_row[self.data_columns]

            if self.create_opt_request.db_type == "hsql":
                terminated, meta = oh.query_if_job_done_direct(self.job_id)
            else:
                terminated, meta = oh.query_if_job_done(self.job_id)

            if terminated:
                print("JobId: ", self.job_id, " job is done")

            if action[0] == cur_row['concurrency'].iloc[-1] and action[1] == cur_row['parallelism'].iloc[-1]:
                break

            time.sleep(1)
        # this reward is of the last influx column which is mapped to that
        # observation so this is the past time steps not the current actions rewards
        return observation, reward, terminated, None, None

    """
    So right now this does not launch another job. It simply resets the observation space to the last jobs influx entries
    Should only be called if there was a termination.
    """

    def reset(self, seed=None, options={'launch_job': False}):
        if options['launch_job']:
            if self.create_opt_request.db_type == "hsql":
                first_meta_data = oh.query_batch_job_direct(self.job_id)
            else:
                first_meta_data = oh.query_job_batch_obj(self.job_id)
            print("InfluxEnv: relaunching job: ", first_meta_data['jobParameters'])
            oh.submit_transfer_request(first_meta_data, optimizer="DDPG")
            self.job_id = int(self.job_id) + 1
            # time.sleep(10)

        if len(self.past_actions) > 0 and len(self.past_rewards) > 0:
            print("Avg reward: ", np.mean(self.past_rewards))
            print("Actions to Count: ", np.unique(self.past_actions, return_counts=True))

        self.past_actions.clear()
        self.past_rewards.clear()
        # env launches the last job again.
        newer_df = self.influx_client.query_space("-5m")  # last min is 2 points.
        self.space_df.append(newer_df)
        self.space_df.drop_duplicates(inplace=True)
        self.space_df.dropna(inplace=True)
        obs = self.space_df[self.data_columns].tail(n=1)
        # will implement a job difficulty score and then based on that run harder jobs and update based on Agent performance
        return obs, {}

    """
    So this will support 3 writing modes:
    none: there is no rendering done
    graph: writes graphs of the df information and past actions and rewards
    ansi: prints the df information to standard out.
    """

    def render(self, mode="None"):
        print("Viewing the graphs on influxdb is probably best but here is stuff from the last epoch")
        print("Rewards: thrpt*rtt", self.past_rewards)
        print("Data")
        if mode == "graph":
            plot = self.space_df.plot(columns=self.data_columns)
            fig = plot.get_figure()
            plt_file_name = '../plots/' + self.create_opt_request.node_id + "_" + str(self.job_id) + str('_.png')
            fig.savefig(plt_file_name)

    """
    Closes in the influx client behind the scenes
    """

    def close(self):
        self.influx_client.close_client()
        self.space_df = None
