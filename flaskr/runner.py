import gymnasium.spaces
import numpy
import pandas
import torch
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter

from .ods_env import ods_influx_gym_env
from .ods_env.ods_rewards import ArslanReward

from flaskr import classes

from .algos.ddpg import agents as ddpg_agents
from .algos.bdq import agents as bdq_agents
from .algos.global_memory import ReplayBuffer as ReplayBufferBDQ

from .algos.ddpg import memory
from .algos.ddpg import utils
from .ods_env.env_utils import smallest_throughput_rtt

from abc import ABC, abstractmethod

enable_tensorboard = os.getenv("ENABLE_TENSORBOARD", default='False').lower() in ('true', '1', 't')
if enable_tensorboard:
    writer = SummaryWriter()

class AbstractTrainer(ABC):
    def warm_buffer(self):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def set_create_request(self, create_opt_req):
        pass

    @abstractmethod
    def close(self):
        pass


class Trainer(object):
    def __init__(self) -> None:
        print("Trainer: use Construct() instead")

    @staticmethod
    def construct(create_req: classes.CreateOptimizerRequest, **kwargs) -> AbstractTrainer:
        optimizer_type = create_req.optimizerType
        # vda2c = "VDA2C"
        # bo = "BO"
        # maddpg = "MADDPG"
        ddpg = "DDPG"
        bdq = "BDQ"

        if optimizer_type == ddpg:
            return DDPGTrainer(create_opt_request=create_req, **kwargs)

        elif optimizer_type == bdq:
            return BDQTrainer(create_req, **kwargs)


def fetch_df(env: ods_influx_gym_env.InfluxEnv, obs_cols: list) -> pandas.DataFrame:
    df = env.influx_client.query_space(time_window="-1d")
    df = df[obs_cols]
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # create diff-drop-in column inplace
    df.insert(0, 'diff_dropin', df['dropin'].diff(periods=1).fillna(0))

    return df


def convert_to_action(par, params_to_actions) -> int:
    """
    Converts influx parallelism or concurrency to action of BDQ agent.
    If not a perfect fit, attempts to approximate through rounding of logs.
    Params:
            par = parameter value to convert
            params_to_actions = dictionary to convert to action
    Returns: BDQ agent action.
    """

    par = int(par)
    if par in params_to_actions:
        # return params_to_actions[par]
        return params_to_actions[min(2, par)]
    else:
        return min(2, int(np.round(np.log2(par))))


def load_clean_norm_dataset(path: str) -> pandas.DataFrame:
    df = pandas.read_csv(path)

    df_pivot = pandas.pivot_table(df, index='_time', columns='_field', values='_value', aggfunc=np.sum)
    try:
        df_pivot = df_pivot.drop(['_field', 'string', 'true'], axis=1)
    except:
        pass
    df_pivot.columns.rename(None, inplace=True)

    for c in df_pivot.columns:
        df_pivot[c] = pandas.to_numeric(df_pivot[c], errors='ignore')

    df_pivot = df_pivot.select_dtypes(include=np.number)
    df_pivot = df_pivot.dropna(axis=1, how='all')
    df_final = df_pivot.dropna(axis=0, how='any')

    df_final.insert(0, 'diff_dropin', df_final['dropin'].diff(periods=1).fillna(0))

    return df_final


class BDQTrainer(AbstractTrainer):
    def __init__(self, create_opt_request: classes.CreateOptimizerRequest, max_episodes=100, batch_size=64):
        self.total_obs_cols = ['active_core_count', 'allocatedMemory',
                               'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min',
                               'dropin', 'dropout', 'packet_loss_rate', 'chunkSize', 'concurrency',
                               'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                               'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']

        self.obs_cols = ['allocatedMemory', 'cpu_frequency_current', 'dropin', 'dropout', 'packet_loss_rate',
                         'chunkSize', 'concurrency', 'destination_latency', 'destination_rtt', 'jobSize',
                         'parallelism', 'pipelining', 'read_throughput', 'source_latency',
                         'source_rtt', 'write_throughput']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request, time_window="-1d",
                                                observation_columns=self.obs_cols)
        self.create_opt_request = create_opt_request
        state_dim = self.env.observation_space.shape[0]
        # 1, 2, 4, 8, 16, 32 = Discrete(6)
        # 2, 4, 8, 16, 32 = Discrete(5)
        self.action_space = gymnasium.spaces.Discrete(3)
        action_dim = self.action_space.n
        self.branches = 2
        
        self.batch_size = batch_size
        self.max_episodes = max_episodes

        self.params_to_actions = {
            2: 0,
            4: 1,
            8: 2,
            16: 3,
            32: 4
        }

        self.actions_to_params = list(self.params_to_actions.keys())

        self.agent = bdq_agents.BDQAgent(
            state_dim=state_dim, action_dims=[action_dim, action_dim], device=self.device, num_actions=2,
            decay=0.992
        )

        self.replay_buffer = ReplayBufferBDQ(state_dimension=state_dim, action_dimension=self.branches)

        self.norm_data = load_clean_norm_dataset('data/benchmark_data.csv')
        self.stats = self.norm_data.describe()

        self.warm_buffer()
        self.save_file_name = f"BDQ_{'influx_gym_env'}"
        self.training_flag = False

        if enable_tensorboard:
            print("[INFO] Tensorboard Enabled")
        else:
            print("[INFO] Tensorboard Disabled")

    def warm_buffer(self):
        df = fetch_df(self.env, self.obs_cols)

        # initialize stats here
        means = self.stats.loc['mean']
        stds = self.stats.loc['std']

        # create utility column inplace
        df = df.assign(utility=lambda x: ArslanReward.construct(x, penalty='diff_dropin'))

        for i in range(df.shape[0] - 1):
            current_row = df.iloc[i]
            obs = current_row[self.obs_cols]

            params = obs[['parallelism', 'concurrency']]
            action = [convert_to_action(p, self.params_to_actions) for p in params]

            next_row = df.iloc[i + 1]
            next_obs = next_row[self.obs_cols]
            
            reward = ArslanReward.compare(current_row['utility'], next_row['utility'], pos_rew=300, neg_rew=-600)
            terminated = False

            norm_obs = (obs.to_numpy() - means[self.obs_cols].to_numpy()) / (stds[self.obs_cols].to_numpy() + 1e-3)
            norm_next_obs = (next_obs.to_numpy() - means[self.obs_cols].to_numpy()) / (stds[self.obs_cols].to_numpy() + 1e-3)

            # norm_obs = obs
            # norm_next_obs = next_obs

            self.replay_buffer.add(norm_obs, action, norm_next_obs, reward, terminated)

        print("Finished Warming Buffer: size=", self.replay_buffer.size)

    def train(self, max_episodes=1000, launch_job=False):
        self.training_flag = True

        episode_rewards = []
        lj = launch_job
        options = {'launch_job': lj}
        print("Before the first env reset()")
        obs = self.env.reset(options=options)[0]  # gurantees job is running
        print("State in train(): ", obs, "\t", "type: ", type(obs))
        obs = np.asarray(a=obs, dtype=numpy.float64)
        lj = False  # previous line launched a job
        # episode = 1 transfer job

        means = self.stats.loc['mean']
        stds = self.stats.loc['std']

        ts = 0
        for episode in range(max_episodes):
            episode_reward = 0
            terminated = False

            print("BDQTrainer.train(): starting episode", episode+1)

            if enable_tensorboard:
                start_stamp = time.time()
            while not terminated:
                time.sleep(5)
                actions = self.agent.select_action(np.array(obs))
                # action = np.clip(action, 1, 32)
                params = [self.actions_to_params[a] for a in actions]

                new_obs, reward, terminated, truncated, info = self.env.step(params, reward_type='arslan')
                ts += 1

                norm_obs = (obs - means[self.obs_cols].to_numpy()) / \
                           (stds[self.obs_cols].to_numpy() + 1e-4)
                norm_next_obs = (new_obs - means[self.obs_cols].to_numpy()) / \
                    (stds[self.obs_cols].to_numpy() + 1e-3)

                # norm_obs = obs
                # norm_next_obs = new_obs

                self.replay_buffer.add(norm_obs, actions, norm_next_obs, reward, terminated)
                obs = new_obs

                if self.replay_buffer.size > self.batch_size:
                    self.agent.train(replay_buffer=self.replay_buffer, batch_size=self.batch_size)

                episode_reward += reward

            if terminated:
                if enable_tensorboard:
                    time_elapsed = time.time() - start_stamp
                    writer.add_scalar("Train/ep_reward", episode_reward, episode)
                    writer.add_scalar("Train/job_time", time_elapsed, episode)
                    
                if (episode+1) < max_episodes:
                    obs = self.env.reset(options={'launch_job': True})[0]
                    obs = np.asarray(a=obs, dtype=numpy.float64)

                print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)

        self.training_flag = False
        self.agent.save_checkpoint("influx_gym_env")
        self.env.render(mode="graph")
        self.training_flag = False
        print("BDQTrainer.train(): THREAD EXITING")

        if enable_tensorboard:
            writer.flush()
            writer.close()
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

    def close(self):
        self.env.close()


class DDPGTrainer(AbstractTrainer):
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
        self.create_opt_request = create_opt_request  # this gets updated every call
        self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request, time_window="-1d",
                                                observation_columns=self.obs_cols)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = ddpg_agents.DDPGAgent(
            state_dim=state_dim, action_dim=action_dim, device=self.device, max_action=None
        )

        self.replay_buffer = memory.ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
        self.warm_buffer()
        self.save_file_name = f"DDPG_{'influx_gym_env'}"
        try:
            os.mkdir('./models')
        except Exception as e:
            pass
        self.training_flag = False

    # Lets say we select the past 10 jobs worth of data what happens?
    def warm_buffer(self):
        print("warming buffer")
        df = self.env.influx_client.query_space(time_window="-1d")
        df = df[self.obs_cols]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        for i in range(df.shape[0] - 1):
            current_row = df.iloc[i]
            obs = current_row[self.obs_cols]
            action = obs[['concurrency', 'parallelism']]
            next_obs = df.iloc[i + 1]
            if next_obs['write_throughput'] < next_obs['read_throughput']:
                thrpt = next_obs['write_throughput']
                rtt = next_obs['destination_rtt']
            else:
                thrpt = next_obs['read_throughput']
                rtt = next_obs['source_rtt']
            reward = rtt * thrpt
            terminated = False
            self.replay_buffer.add(obs, action, next_obs, reward, terminated)
        print("Finished Warming Buffer: size=", self.replay_buffer.size)

    def train(self, max_episodes=2, batch_size=100, launch_job=False):
        self.training_flag = True
        episode_rewards = []
        lj = launch_job
        options = {'launch_job': lj}
        print("Before the first env reset()")
        obs = self.env.reset(options=options)[0]  # gurantees job is running
        print("State in train(): ", obs, "\t", "type: ", type(obs))
        obs = np.asarray(a=obs, dtype=numpy.float64)
        lj = False  # previous line launched a job
        # episode = 1 transfer job
        ts = 0
        for episode in range(max_episodes):
            episode_reward = 0
            terminated = False
            while not terminated:
                action = (self.agent.select_action(np.array(obs)))
                action = np.rint(action)
                action = np.clip(action, 1, 32)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                ts += 1
                self.replay_buffer.add(obs, action, new_obs, reward, terminated)
                obs = new_obs
                if self.replay_buffer.size > batch_size:
                    if ts % 10 == 0:
                        self.agent.train(replay_buffer=self.replay_buffer, batch_size=batch_size)

                episode_reward += reward

            if terminated:
                if (episode+1) < max_episodes:
                    obs = self.env.reset(options={'launch_job': True})[0]
                    obs = np.asarray(a=obs, dtype=numpy.float64)

                print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)

        self.training_flag = False
        self.agent.save_checkpoint("influx_gym_env")
        self.env.render(mode="graph")
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

    def close(self):
        self.env.close()
