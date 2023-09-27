import time

import gym
from gym import spaces
import numpy as np
import pandas as pd
from flaskr.classes import CreateOptimizerRequest
import flaskr.ods_env.ods_helper as oh
from flaskr.ods_env import env_utils
from flaskr.ods_env.influx_query import InfluxData
from flaskr.algos.ddpg.memory import ReplayBuffer
from .ods_rewards import DefaultReward, ArslanReward, JacobReward
from .ods_rewards import RatioReward
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


def construct_reward():
    pass


class PIDEnv(gym.Env):

    def __init__(self, create_opt_req: CreateOptimizerRequest, target_thput, config=None,
                 action_space_discrete=False, render_mode=None, time_window="-2m", observation_columns=[],
                 host_url=None):
        super(PIDEnv, self).__init__()
        self.host_url = host_url
        self.replay_buffer = None
        self.create_opt_request = create_opt_req
        print(create_opt_req.node_id.split("-"))
        bucket_name = create_opt_req.node_id.split("-")[0]
        print(create_opt_req.node_id)
        self.influx_client = InfluxData(bucket_name=bucket_name, transfer_node_name=create_opt_req.node_id,
                                        file_name=None, time_window=time_window)
        # gets last 7 days worth of data. Gonna be a lil slow to create
        self.space_df = self.influx_client.query_space(time_window)
        self.job_id = create_opt_req.job_id
        if len(observation_columns) > 0:
            self.data_columns = observation_columns
        else:
            self.data_columns = self.space_df.columns.values

        self.state_columns = ['parallelism', 'concurrency']

        if action_space_discrete:
            # self.action_space = spaces.Discrete(3)  # drop stay increase
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,))  # for now cc, p only
        # So this is probably not defined totally properly as I assume dimensions are 0-infinity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.past_rewards = []
        self.past_actions = []
        self.render_mode = render_mode

        self.drop_in = 0
        self.past_utility = 0

        self.target_thput = target_thput
        self.target_freq = 1000.  # MHz
        self.freq_max = 4000.

        self.error_0 = 0.
        self.dt_0 = 50.

        self.error_1 = 0.
        self.dt_1 = 50.

        self.error_2 = 0.
        self.dt_2 = 50.

        self.mix = 0.
        if config is not None:
            self.mix = config['pid_frequency_mix']

        self.max_par = 16

        self.max_err = (1. - self.mix) * self.target_thput + (self.mix * 3000.)
        self.max_sum = 300 * self.max_err
        self.max_diff = (0.75 * self.max_err) / 4

        print("[DEBUG/PID_ENV]", self.host_url)
        print("Finished constructing the PID Gym Env")

    def set_target_thput(self, target_thput):
        """
        This function recalculates the max err, integral and derivative
        """
        print("[PID] Setting target thput to", str(target_thput))
        self.target_thput = target_thput

        self.max_err = (1. - self.mix) * self.target_thput + self.mix * (self.freq_max - self.target_freq)
        self.max_sum = 300 * self.max_err
        self.max_diff = (0.75 * self.max_err) / 4

    def set_target_freq(self, target_freq, freq_max):
        """
        This function recalculates max err, integral and derivative
        """
        print("[PID] Setting target frequency to [", str(target_freq), str(freq_max), "]")
        self.target_freq = target_freq
        self.freq_max = freq_max

        self.max_err = (1. - self.mix) * self.target_thput + self.mix * (self.freq_max - self.target_freq)
        self.max_sum = 300 * self.max_err
        self.max_diff = (0.75 * self.max_err) / 4

    """
    Stepping the env and a live transfer behind the scenes.
    Params: action = {"concurrency":1-64, "parallelism":1-64, "pipelining:1-100}
            reward_type = None | 'default' | 'arslan' (default: None)
    Returns: observation: last non-na entry from influx which contains the action for that obs
             reward: the throughput * rtt of the slowest link(source or destination)
             done: the official status if this job is done. If it is done than we need to reset() which should launch a job?
             info: Currently None and not necessary I believe.
    """

    def step(self, action: list, reward_type=None):
        conc_nan = True
        para_nan = True

        last_row = None
        while conc_nan or para_nan:
            df = self.influx_client.query_space("-2m")
            self.space_df = pd.concat([self.space_df, df])
            last_row = self.space_df.tail(n=1)
            observation = last_row[self.data_columns]
            if self.create_opt_request.db_type == "hsql":
                terminated, _ = oh.query_if_job_done_direct(self.job_id, ts_url=self.host_url)
            else:
                terminated, _ = oh.query_if_job_done(self.job_id)

            conc_nan = last_row['concurrency'].isna().any()
            para_nan = last_row['parallelism'].isna().any()
            time.sleep(2)

        # submit action to take with TS
        if int(last_row['concurrency'].iloc[0]) != action[0] or int(last_row['parallelism'].iloc[0]) != action[1]:
            oh.send_application_params_tuple(
                transfer_node_name=self.create_opt_request.node_id,
                cc=action[0], p=action[1], pp=1, chunkSize=0
            )
        else:
            # introduce delay to prevent param spam
            print("[PID] Duplicate; PID sleeping")
            time.sleep(10)

        # block until action applies to TS
        terminated = False
        reward = 0
        count = 0
        fail_count = 0

        start_time = time.time()

        while True:
            # print("Blocking till action: ", action)
            df = self.influx_client.query_space("-30s")
            if not set(self.data_columns).issubset(df.columns):
                time.sleep(1)
                continue
            # For every query we want to add to the buffer bc its more data on transfer.
            for i in range(df.shape[0]):
                current_row = df.iloc[i]
                obs = current_row[self.data_columns]
                obs_action = obs[['parallelism', 'concurrency']]

                if obs_action['concurrency'] == action[0] and obs_action['parallelism'] == action[1]:
                    count += 1
                else:
                    fail_count += 1

                if fail_count % 10 == 0:
                    oh.send_application_params_tuple(
                        transfer_node_name=self.create_opt_request.node_id,
                        cc=action[0], p=action[1], pp=1, chunkSize=0
                    )
                    fail_count = 0

            if self.create_opt_request.db_type == "hsql":
                terminated, _ = oh.query_if_job_done_direct(self.job_id, ts_url=self.host_url)
            else:
                terminated, _ = oh.query_if_job_done(self.job_id)

            if count >= 2 or terminated: break
            time.sleep(2)

        dt = time.time() - start_time
        last_row = df.tail(n=1)
        all_observation = last_row[self.data_columns]

        err = self.target_thput - all_observation['read_throughput'].to_numpy()[-1]
        err = (1. - self.mix) * err + self.mix * (all_observation['cpu_frequency_current'] - self.target_freq)

        err = err.to_numpy()[0]
        err_sum = self.error_1 + (err * dt)
        err_diff = (err - self.error_2) / dt

        self.error_2 = self.error_1
        self.error_1 = self.error_0
        self.error_0 = err

        self.dt_2 = self.dt_1
        self.dt_1 = self.dt_0
        self.dt_0 = dt

        # err = err - self.error_1
        # err_diff = err_diff + (self.error_2 / self.dt_2) - 2 * (self.error_1 / self.dt_1)

        err /= self.max_err
        err_sum /= self.max_sum
        err_diff /= self.max_diff

        reward = 1. - err

        observation = all_observation[['parallelism', 'concurrency']] / self.max_par
        # observation = observation / self.max_par

        # observation.loc[:, 'parallelism'] = observation.loc[:, 'parallelism'] / self.max_par
        # observation.loc[:, 'concurrency'] = observation.loc[:, 'concurrency'] / self.max_par

        # print(err, err_sum, err_diff)
        observation.insert(0, "err", err)
        observation.insert(1, "err_sum", err_sum)
        observation.insert(2, "err_diff", err_diff)

        self.past_rewards.append(reward)

        # if terminated:
        #     print("JobId: ", self.job_id, " job is done")

        # print(observation)
        info = {
            'nic_speed': last_row['nic_speed'].to_numpy()[0],
            'read_throughput': last_row['read_throughput'].to_numpy()[0],
            'cpu_frequency_min': last_row['cpu_frequency_min'].to_numpy()[0],
            'cpu_frequency_max': last_row['cpu_frequency_max'].to_numpy()[0],
            'cpu_frequency_current': last_row['cpu_frequency_current'].to_numpy()[0]
        }

        return observation, reward, terminated, None, info

    """
    So right now this does not launch another job. It simply resets the observation space to the last jobs influx entries
    Should only be called if there was a termination.
    """

    def reset(self, seed=None, options={'launch_job': False}, optimizer="BDQ"):
        self.past_utility = self.past_utility / 4
        if options['launch_job']:
            if self.create_opt_request.db_type == "hsql":
                first_meta_data = oh.query_batch_job_direct(self.job_id)
            else:
                first_meta_data = oh.query_job_batch_obj(self.job_id)
            # print("InfluxEnv: relaunching job: ", first_meta_data['jobParameters'])
            oh.submit_transfer_request(first_meta_data, optimizer="BDQ")
            time.sleep(10)

        # if len(self.past_actions) > 0 and len(self.past_rewards) > 0:
        #     print("Avg reward: ", np.mean(self.past_rewards))
        #     print("Actions to Count: ", np.unique(self.past_actions, return_counts=True))

        self.past_actions.clear()
        self.past_rewards.clear()
        # env launches the last job again.
        newer_df = self.influx_client.query_space("-5m")  # last min is 2 points.
        self.space_df = pd.concat([self.space_df, newer_df])
        self.space_df.drop_duplicates(inplace=True)
        obs = self.space_df[self.state_columns].tail(n=1)

        err = 0  # self.target_thput - obs['read_throughput'].to_numpy()[-1]
        err_sum = err
        err_diff = err

        if np.isnan(err_sum):
            self.error_1 = 0.
        else:
            self.error_1 = err_sum

        if np.isnan(err):
            self.error_2 = 0.
        else:
            self.error_2 = err

        self.error_0 = 0.
        self.dt_0 = 50.
        self.dt_1 = 50.
        self.dt_2 = 50.

        obs = obs[['parallelism', 'concurrency']]

        # err /= self.max_err
        # err_sum /= self.max_sum
        # err_diff /= self.max_diff

        obs['parallelism'] = obs['parallelism'] / self.max_par
        obs['concurrency'] = obs['concurrency'] / self.max_par

        obs.insert(0, "err", err)
        obs.insert(1, "err_sum", err_sum)
        obs.insert(2, "err_diff", err_diff)

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

        # this below will need debugging
        if mode == "graph":
            # plot = self.space_df.plot(columns=self.data_columns)
            plot = self.space_df.plot(ylabel=self.data_columns)
            fig = plot.get_figure()
            # plt_file_name = '../plots/' + self.create_opt_request.node_id + "_" + str(self.job_id) + str('_.png')
            # fig.savefig(plt_file_name)

    """
    Closes in the influx client behind the scenes
    """

    def close(self):
        self.influx_client.close_client()
        self.space_df = None

    def set_buffer(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    def fill_buffer(self, df: pd.DataFrame):
        for i in range(df.shape[0] - 1):
            current_row = df.tail(n=1)

            obs = current_row[self.data_columns]
            action = obs[['parallelism', 'concurrency']]
            next_obs = df.iloc[i + 1]
            thrpt, rtt = env_utils.smallest_throughput_rtt(last_row=current_row)
            reward_params = JacobReward.Params(
                throughput=thrpt,
                rtt=rtt,
                c=current_row['concurrency'].iloc[-1],
                max_cc=self.create_opt_request.max_concurrency,
                p=current_row['parallelism'].iloc[-1],
                max_p=self.create_opt_request.max_parallelism,
            )
            reward = JacobReward.calculate(reward_params)
            # current_job_id = current_row['jobId']
            terminated = False
            # if next_obs['jobId'] != current_job_id:
            #     terminated = True

            self.replay_buffer.add(obs, action, next_obs, reward, terminated)
