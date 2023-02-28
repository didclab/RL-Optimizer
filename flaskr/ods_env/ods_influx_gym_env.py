import gym
from gym import spaces
import numpy as np
import pandas
from flaskr.classes import CreateOptimizerRequest
import flaskr.ods_env.ods_helper as oh
from flaskr.ods_env import env_utils
from flaskr.ods_env.influx_query import InfluxData

headers = {"Content-Type": "application/json"}

# display all the  rows
pandas.set_option('display.max_rows', None)

# display all the  columns
pandas.set_option('display.max_columns', None)

# set width  - 100
pandas.set_option('display.width', 100)

# set column header -  left
pandas.set_option('display.colheader_justify', 'left')

# set precision - 5
pandas.set_option('display.precision', 5)

class InfluxEnv(gym.Env):

    def __init__(self, create_opt_req, reward_function=lambda rtt, thrpt: (rtt * thrpt),
                 action_space_discrete=False, render_mode=None, time_window="-2m", observation_columns=[]):
        super(InfluxEnv, self).__init__()
        self.create_opt_request = create_opt_req
        print(create_opt_req.node_id.split("-"))
        bucket_name = create_opt_req.node_id.split("-")[0]
        self.influx_client = InfluxData(bucket_name=bucket_name, transfer_node_name=create_opt_req.node_id,
                                        file_name=None, time_window=time_window)
        # gets last 7 days worth of data. Gonna be a lil slow to create
        print("querying the space df")
        self.space_df = self.influx_client.query_space(time_window)
        self.job_id = create_opt_req.job_id
        if len(observation_columns) > 0:
            self.data_columns = observation_columns
        else:
            self.data_columns = self.space_df.columns.values
        print("obs space columns are:", self.data_columns)
        self.reward_function = reward_function  # this function is then used to evaluate the obs space of the last entry
        if action_space_discrete:
            self.action_space = spaces.Discrete(3)  # drop stay increase
        else:
            self.action_space = spaces.Box(low=1, high=32, shape=(3,))
            # So this is probably not defined totally properly as I assume dimensions are 0-infinity
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.data_columns),))
        self.past_rewards = []
        self.past_actions = []
        self.render_mode = render_mode
        self.past_job_ids = []
        print("Finished constructing the Influx Gym Env")

    """
    Stepping the env and a live transfer behind the scenes.
    Params: action = {"concurrency":1-64, "parallelism":1-64, "pipelining:1-100}
    Returns: observation: last non-na entry from influx which contains the action for that obs
             reward: the throughput * rtt of the slowest link(source or destination)
             done: the official status if this job is done. If it is done than we need to reset() which should launch a job?
             info: Currently None and not necessary I believe.
    """

    def step(self, action):
        if not action.all():
            return {}, {}, {}, {}
        print("Step: action:", action)
        self.past_actions.append(action)
        self.past_actions.append(action)

        newer_df = self.influx_client.query_space("-5m")  # last min is 2 points.
        self.space_df.append(newer_df)
        self.space_df.drop_duplicates(inplace=True)
        self.space_df.dropna(inplace=True)

        # Need to loop while we have not gotten the next observation of the agents.
        last_row = self.space_df.tail(n=1)
        observation = last_row[self.data_columns]
        print("Observation: \n", observation)

        # Here we need to use monitoring API to know when the job is formally done vs when its running very slow
        if 'jobId' in last_row:
            terminated, _ = oh.query_if_job_done(last_row['jobId'])
        else:
            terminated, _ = oh.query_if_job_done(self.create_opt_request.job_id)

        thrpt, rtt = env_utils.smallest_throughput_rtt(last_row=last_row)

        if action[0] < 1 or action[1] < 1 or action[2]< 1:
            reward = -100
        else:
            reward = self.reward_function(rtt, thrpt)
            oh.send_application_params_tuple(action[0], action[1], action[2], 0)

        # this reward is of the last influx column which is mapped to that observation so this is the past time steps not the current actions rewards
        self.past_rewards.append(reward)
        return observation, reward, terminated, None

    """
    So right now this does not launch another job. It simply resets the observation space to the last jobs influx entries
    """

    def reset(self, seed=None, options={'launch_job': False}):
        print("Past Actions: ", self.past_actions)
        print("Past Rewards: ", self.past_rewards)
        self.past_actions.clear()
        self.past_rewards.clear()
        # env launches the last job again.
        newer_df = self.influx_client.query_space("-5m")  # last min is 2 points.
        self.space_df.append(newer_df)
        self.space_df.drop_duplicates(inplace=True)
        self.space_df.dropna(inplace=True)
        print("Reset(): space_df shape:", self.space_df.shape)
        obs = self.space_df[self.data_columns].tail(n=1)
        print("Reset(): obs shape: ", obs.shape)
        # will implement a job difficulty score and then based on that run harder jobs and update based on Agent performance
        if options['launch_job']:
            first_meta_data = oh.query_job_batch_obj(self.create_opt_request.job_id)
            oh.submit_transfer_request(first_meta_data)
            # Here I would want to compute the transfers difficulty which we could measure
            self.past_job_ids.append(self.job_id)
            self.job_id += 1

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
