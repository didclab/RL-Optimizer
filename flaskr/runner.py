import gymnasium.spaces
import numpy
import pandas
# import pandas as pd
import torch
import numpy as np
import os
import time
import json

# from collections import deque

from torch.utils.tensorboard import SummaryWriter
from .ods_env import env_utils
from .ods_env import ods_influx_gym_env
from .ods_env import ods_pid_env
from .ods_env.gan_env import ConvGeneratorNet
from .ods_env.ods_rewards import ArslanReward, RatioReward, JacobReward

from flaskr import classes

from .algos.ddpg import agents as ddpg_agents
from .algos.bdq import agents as bdq_agents
from .algos.global_memory import ReplayBuffer as ReplayBufferBDQ

from .algos.ddpg import memory
from .algos.ddpg import utils
from .ods_env.env_utils import smallest_throughput_rtt

from abc import ABC, abstractmethod
from tqdm import tqdm

enable_tensorboard = os.getenv("ENABLE_TENSORBOARD", default='False').lower() in ('true', '1', 't')
writer = None
if enable_tensorboard:
    writer = SummaryWriter('./runs/deploy_pid3_bdq/')

class AbstractTrainer(ABC):
    def __init__(self, trainer_type):
        self.trainer_type = trainer_type

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

    def parse_config(self, json_file="config/default.json"):
        with open(json_file, 'r') as f:
            configs = json.load(f)

        return configs


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
        return params_to_actions[min(3, par)]
    else:
        return min(3, int(np.round(np.log2(par))))


def load_clean_norm_dataset(path: str) -> pandas.DataFrame:
    # df = pandas.read_csv(path)
    df_pivot = pandas.read_csv(path)

    # df_pivot = pandas.pivot_table(df, index='_time', columns='_field', values='_value', aggfunc=np.sum)
    try:
        df_pivot = df_pivot.drop(['_field', 'string', 'true'], axis=1)
    except:
        pass
    # df_pivot.columns.rename(None, inplace=True)

    for c in df_pivot.columns:
        df_pivot[c] = pandas.to_numeric(df_pivot[c], errors='ignore')

    df_pivot = df_pivot.select_dtypes(include=np.number)
    df_pivot = df_pivot.dropna(axis=1, how='all')
    df_final = df_pivot.dropna(axis=0, how='any')

    # df_final.insert(0, 'diff_dropin', df_final['dropin'].diff(periods=1).fillna(0))

    return df_final
    # return df_pivot


class BDQTrainer(AbstractTrainer):
    def __init__(self, create_opt_request: classes.CreateOptimizerRequest, max_episodes=100, batch_size=64, config_file='config/default.json'):
        super().__init__("BDQ")

        self.config = self.parse_config(config_file)

        self.total_obs_cols = ['active_core_count', 'allocatedMemory',
                               'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min',
                               'dropin', 'dropout', 'packet_loss_rate', 'chunkSize', 'concurrency',
                               'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                               'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']

        self.obs_cols = ['allocatedMemory', 'cpu_frequency_current', 'dropin', 'dropout', 'packet_loss_rate',
                         'chunkSize', 'concurrency', 'destination_latency', 'destination_rtt', 'jobSize',
                         'parallelism', 'pipelining', 'read_throughput', 'source_latency',
                         'source_rtt', 'write_throughput']

        self.obs_cols_pid = ['parallelism', 'concurrency', 'err', 'err_sum', 'err_diff']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_pid_env = self.config['use_pid']
        if self.use_pid_env:
            print("[INFO] Using PID Environment")

        if self.use_pid_env:
            self.obs_cols = self.obs_cols_pid
            env_cols = ['concurrency', 'parallelism', 'read_throughput']
            self.env = ods_pid_env.PIDEnv(create_opt_req=create_opt_request, time_window="-1d",
                                          observation_columns=env_cols, target_thput=819.,
                                          action_space_discrete=True)
        else:
            self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request, time_window="-1d",
                                                    observation_columns=self.obs_cols)

        self.create_opt_request = create_opt_request
        state_dim = self.env.observation_space.shape[0]
        # 1, 2, 4, 8, 16, 32 = Discrete(6)
        # 2, 4, 8, 16, 32 = Discrete(5)
        self.action_space = gymnasium.spaces.Discrete(4)
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
            decay=0.992, writer=writer
        )
        # self.history = deque(maxlen=5)

        self.replay_buffer = ReplayBufferBDQ(state_dimension=state_dim, action_dimension=self.branches)

        self.norm_data = load_clean_norm_dataset(self.config['data'])
        self.stats = self.norm_data.describe()
        # self.df_columns = self.norm_data.columns
        # print("[NORM DATA] number of columns:", len(self.df_columns))

        self.use_ratio = True
        self.pretrain_mode = self.config['pretrain']
        self.eval_mode = self.config['eval']
        self.deploy_mode = self.config['deploy']

        self.deploy_ctr = 0
        self.deploy_job_ctr = 0

        if self.eval_mode or self.deploy_mode:
            self.agent.load_checkpoint(self.config['checkpoint'])
            mode_str = "Eval" if self.eval_mode else "Deploy"
            print("[INFO] In {0} Mode".format(mode_str))

        if self.pretrain_mode:
            print("[INFO] In Rapid Pretrain")

            if self.config['use_gan']:
                self.gen = ConvGeneratorNet(100).double().to(self.device)
                self.gen.load_state_dict(torch.load(self.config['gan_path']))
                self.gan.eval()
                print("[GAN] Generator Loaded")

        if not self.eval_mode and not self.config['use_gan'] and not self.use_pid_env:
            self.warm_buffer()

        if self.config['savefile_name']:
            self.save_file_name = self.config['savefile_name']
        else:
            self.save_file_name = f"BDQ_{'influx_gym_env'}"
        self.training_flag = False

        if enable_tensorboard:
            print("[INFO] Tensorboard Enabled")
        else:
            print("[INFO] Tensorboard Disabled")

    def warm_buffer(self):
        if self.pretrain_mode:
            # data = load_clean_norm_data('data/pivoted_data.csv')
            df = self.norm_data
        else:
            df = fetch_df(self.env, self.obs_cols)

        # initialize stats here
        means = self.stats.loc['mean']
        stds = self.stats.loc['std']

        # create utility column inplace
        if not self.use_ratio and not self.use_pid_env:
            df = df.assign(utility=lambda x: ArslanReward.construct(x, penalty='diff_dropin', bwidth=0.1))

        for i in range(0, df.shape[0] - 1, 1):
            current_row = df.iloc[i]
            obs = current_row[self.obs_cols]

            next_row = df.iloc[i + 1]
            next_obs = next_row[self.obs_cols]

            params = next_obs[['parallelism', 'concurrency']]
            action = [convert_to_action(p, self.params_to_actions) for p in params]

            if self.use_ratio:
                reward = RatioReward.construct(next_obs)
            else:
                reward = ArslanReward.compare(current_row['utility'], next_row['utility'], pos_rew=300, neg_rew=-600)
            terminated = False

            if self.use_pid_env:
                norm_obs = obs
                norm_next_obs = next_obs
            else:
                norm_obs = (obs.to_numpy() - means[self.obs_cols].to_numpy()) / (stds[self.obs_cols].to_numpy() + 1e-3)
                norm_next_obs = (next_obs.to_numpy() - means[self.obs_cols].to_numpy()) / (stds[self.obs_cols].to_numpy() + 1e-3)

            # norm_obs = obs
            # norm_next_obs = next_obs

            self.replay_buffer.add(norm_obs, action, norm_next_obs, reward, terminated)

        print("Finished Warming Buffer: size=", self.replay_buffer.size)


    def rapid_pretrain(self, max_episodes=2000000):
        """
        The purpose of this function is to bypass the environment
        entirely and to instead use a surrogate offline data or Generator
        to rapidly generate (state, next_state) for the agent's replay
        buffer or replace the replay buffer entirely.

        This should allow for rapid prototyping and for the agent to
        quickly learn or be pre-trained for a real network. This will
        NOT interact with the real environment.

        NOTE: Arslan's reward won't work here. Assumes rewards that
        can be computed at any time.
        """
        self.training_flag = True

        # print("Before surrogate used")

        # episode = 1 transfer job
        cols = ['active_core_count', 'allocatedMemory', 'avgFileSize',
                'bytesDownloaded', 'bytesUploaded', 'bytes_recv', 'bytes_sent',
                'chunkSize', 'concurrency', 'cpu_frequency_current',
                'cpu_frequency_max', 'cpu_frequency_min', 'destination_latency',
                'destination_rtt', 'dropin', 'dropout', 'errin', 'errout', 'freeMemory',
                'jobSize', 'latency', 'maxMemory', 'memory', 'nic_mtu',
                'packet_loss_rate', 'packets_recv', 'packets_sent', 'parallelism',
                'pipelining', 'read_throughput', 'rtt', 'source_latency', 'source_rtt',
                'write_throughput']

        means = self.stats.loc['mean'][cols].to_numpy()
        stds = self.stats.loc['std'][cols].to_numpy()

        ts = 0
        reward_type = 'ratio'

        """
        Generator replaces the replay buffer. Generated entries need other
        information filled in. In particular:
        - Reward
        - Done/Terminated/Truncated (for now, this is always false)
        - Action (this must be de-normed and then discretized if needed)

        This has been put on hold for now.
        """

        use_gan = self.config['use_gan']
        for it in tqdm(range(max_episodes), unit='it', total=max_episodes):
            episode_reward = 0
            terminated = False

            # print("BDQTrainer.train(): starting episode", episode+1)
            if self.replay_buffer.size > self.batch_size:
                self.agent.train(replay_buffer=self.replay_buffer, batch_size=self.batch_size)

            if use_gan:
                """
                1. Generate noise
                2. Create synthetic data
                3. Denorm next state/obs
                4. Construct action and reward
                """
                noise = torch.randn(256, 100, 1, 1, device=self.device).double()
                with torch.no_grad():
                    gen_tensor = self.gen(noise).squeeze()
                    gen_set = gen_tensor.transpose(1, 0).cpu().numpy()

                norm_obs = gen_set[0]
                norm_next_obs = gen_set[1]

                denorm_set = (norm_next_obs * (stds + 1e-3)) + means
                denorm_df = pandas.Dataframe(denorm_set, columns=cols).round()

                params = denorm_df[['parallelism', 'concurrency']]
                actions = []
                for _, row in denorm_df.iterrows():
                    actions.append([convert_to_action(p, self.params_to_actions) for p in row])

                rewards = RatioReward.construct(denorm_df)

                self.replay_buffer.add_batch(norm_obs, actions, norm_next_obs, rewards, np.repeat(False, 256))



        self.training_flag = False
        self.agent.save_checkpoint(self.save_file_name)
        self.training_flag = False
        # print("BDQTrainer.train(): THREAD EXITING")

        # if enable_tensorboard:
        #     writer.flush()
        #     writer.close()
        return []

    def train(self, max_episodes=1500, launch_job=False):
        if self.deploy_mode:
            return [env_utils.consult_agent(self, writer, self.deploy_job_ctr)]
        elif self.eval_mode:
            return [env_utils.test_agent(self, writer, eval_episodes=15)]
        elif self.pretrain_mode:
            return self.rapid_pretrain()

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
        reward_type = 'arslan'
        if self.use_ratio:
            reward_type = 'ratio'

        action_log = None
        if self.config['log_action']:
            action_log = open("actions_train.log", 'a')

        state_log = None
        if self.config['log_state']:
            state_log = open("state_train.log", 'a')

        best_mean_rew = 0.
        eval_counter = 0
        eval_when = self.config['train_eval_frequency']
        eval_how_many = self.config['train_eval_num']
        temp_file_name = self.config["savefile_name"] + '_temp'
        for episode in tqdm(range(max_episodes), unit='ep'):
            episode_reward = 0
            terminated = False

            # print("BDQTrainer.train(): starting episode", episode+1)
            eval_counter += 1
            if eval_counter == eval_when:
                eval_counter = 0
                # 1. save model to temp
                self.agent.save_checkpoint(temp_file_name)

                # 2. call and wait for eval
                print("Evaluating model at", str(episode))
                mean_reward = self.test_agent(
                    eval_episodes=eval_how_many, use_checkpoint=temp_file_name, use_id=episode
                )

                # 3. if better, save model as best
                if best_mean_rew < mean_reward:
                    best_mean_rew = mean_reward
                    self.agent.save_checkpoint("best_" + self.config['savefile_name'])

                # 4. reset environment
                obs = self.env.reset(options={'launch_job': True})[0]
                obs = np.asarray(a=obs, dtype=numpy.float64)

            if enable_tensorboard:
                start_stamp = time.time()
            while not terminated:
                time.sleep(5)

                if self.use_pid_env:
                    norm_obs = obs
                else:
                    norm_obs = (obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

                # norm_obs = self.history[:3]
                obs_np = np.array(norm_obs)
                # print("[Normalized Obs]", obs_np)

                if state_log:
                    state_log.write(np.array2string(obs_np, precision=3, seperator=',') + '\n')

                # history_len = len(self.history)
                if self.replay_buffer.size <= 1e2: # or history_len < 4:
                    actions = [self.env.action_space.sample(), self.env.action_space.sample()]
                    # print("[Replay Buffer Size]", self.replay_buffer.size)
                else:
                    actions = self.agent.select_action(obs_np)
                # action = np.clip(action, 1, 32)
                params = [self.actions_to_params[a] for a in actions]
                if action_log:
                   action_log.write(str(params) + "\n")

                new_obs, reward, terminated, truncated, info = self.env.step(params, reward_type=reward_type)
                ts += 1

                if self.use_pid_env:
                    norm_next_obs = new_obs
                else:
                    norm_next_obs = (new_obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

                # norm_obs = obs
                # norm_next_obs = new_obs

                # self.history.append(norm_next_obs)

                # if history_len == 4:
                self.replay_buffer.add(norm_obs, actions, norm_next_obs, reward, terminated)
                obs = new_obs

                if self.replay_buffer.size > 1e2:
                    self.agent.train(replay_buffer=self.replay_buffer, batch_size=self.batch_size)

                episode_reward += reward

            if terminated:
                if enable_tensorboard:
                    time_elapsed = time.time() - start_stamp
                    writer.add_scalar("Train/average_reward_step", episode_reward / ts, episode)
                    writer.add_scalar("Train/job_time", time_elapsed, episode)
                    writer.add_scalar("Train/throughput", 32. / time_elapsed, episode)

                ts = 0
                if (episode+1) < max_episodes and eval_counter < eval_when-1:
                    obs = self.env.reset(options={'launch_job': True})[0]
                    obs = np.asarray(a=obs, dtype=numpy.float64)

                # print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)

                if self.replay_buffer.size > 1e2:
                    # print("[Epsilon-Decay] Updating Epsilon")
                    self.agent.update_epsilon()

                self.agent.save_checkpoint(self.config["savefile_name"] + '_temp')

        self.training_flag = False
        self.agent.save_checkpoint(self.config["savefile_name"] + '_final')
        # self.env.render(mode="graph")
        self.training_flag = False
        print("BDQTrainer.train(): THREAD EXITING")

        if self.config['log_state']:
            state_log.flush()
            state_log.close()

        if self.config['log_action']:
            action_log.flush()
            action_log.close()

        if enable_tensorboard:
            writer.flush()
            writer.close()
        return episode_rewards

    def test_agent(self, eval_episodes=10, seed=0, use_checkpoint=None, use_id=None):
        if use_checkpoint is None:
            self.agent.set_eval()

        options = {'launch_job': False}
        state = self.env.reset(options=options)[0]  # gurantees job is running

        episodes_reward = []
        terminated = False

        job_size = 32. # Gb

        means = self.stats.loc['mean']
        stds = self.stats.loc['std']

        reward_type = 'ratio'

        greedy_actions = [3, 2, 1, 1]
        i = 0

        action_log = None
        state_log = None

        if self.config['log_action']:
            action_log = open("actions_eval.log", 'a')
            # if use_id is None:
            #     action_log = open("actions_eval.log", 'a')
            # else:
            #     action_log = open("actions_eval_"+str(use_id)+".log", 'a')

        if self.config['log_state']:
            state_log = open("state_eval.log", 'a')
            # if use_id is None:
            #     state_log = open("states_eval.log", 'a')
            # else:
            #     state_log = open("states_eval_"+str(use_id)+".log", 'a')

        eval_agent = None
        if use_checkpoint is not None:
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.action_space.n

            check_agent = bdq_agents.BDQAgent(
                state_dim=state_dim, action_dims=[action_dim, action_dim], device=self.device, num_actions=2,
                decay=0.992, writer=writer
            )


        start_ep = 0 if use_id is None else use_id
        switch_ep = eval_episodes >> 1
        for ep in tqdm(range(start_ep, eval_episodes, 1), unit='ep'):
            episode_reward = 0

            if ep < switch_ep or use_id is None:
                eval_agent = self.agent
            else:
                eval_agent = check_agent

            if action_log is not None:
                action_log.write("======= Episode " + str(ep) + " =======\n")
            if state_log is not None:
                state_log.write("======= Episode " + str(ep) + " =======\n")

            terminated = False
            ts = 0
            i = 0
            start_stamp = time.time()
            while not terminated:
                if not self.use_pid_env:
                    state = (state - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

                if state_log is not None:
                    state_log.write(np.array2string(np.array(state), precision=3, seperator=',') + '\n')

                with torch.no_grad():
                    ts += 1
                    if self.config['test_greedy']:
                        actions = [greedy_actions[i], 3]
                        i = min(ts, 3)
                    else:
                        actions = eval_agent.select_action(np.array(state), bypass_epsilon=True)

                params = [self.actions_to_params[a] for a in actions]
                next_state, reward, terminated, truncated, info = self.env.step(params, reward_type=reward_type)

                if action_log is not None:
                    action_log.write(str(params) + "\n")

                episode_reward += reward
                state = next_state

            time_elapsed = time.time() - start_stamp
            throughput = job_size / time_elapsed
            episodes_reward.append(episode_reward)

            if enable_tensorboard:
                # writer.add_scalar("Eval/episode_reward", episode_reward, ep)
                writer.add_scalar("Eval/average_reward_step", episode_reward / ts, ep)
                writer.add_scalar("Eval/throughput", throughput, ep)

            if ep < eval_episodes-1:
                i = 0
                state = self.env.reset(options={'launch_job': True})[0]


        # action_log.close()
        if enable_tensorboard:
            writer.flush()
            writer.close()

        if state_log:
            state_log.flush()
            state_log.close()

        if action_log:
            action_log.flush()
            action_log.close()

        return np.mean(episodes_reward)

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
                 update_policy_time_steps=20, config_file='config/default.json'):
        super().__init__("DDPG")
        # self.obs_cols = ['active_core_count', 'allocatedMemory',
        #                  'bytes_recv', 'bytes_sent', 'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min',
        #                  'dropin', 'dropout', 'errin', 'errout', 'freeMemory', 'maxMemory', 'memory',
        #                  'packet_loss_rate', 'packets_recv',
        #                  'packets_sent', 'bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency',
        #                  'destination_latency', 'destination_rtt',
        #                  'jobSize', 'parallelism', 'pipelining', 'read_throughput', 'source_latency', 'source_rtt',
        #                  'write_throughput']
        self.config = self.parse_config(config_file)
        self.obs_cols = ['active_core_count',
                         'dropin', 'dropout', 'packets_recv', 'packets_sent', 'chunkSize', 'concurrency',
                         'destination_latency', 'destination_rtt',
                        'parallelism', 'read_throughput', 'source_latency', 'source_rtt',
                         'write_throughput']

        self.obs_cols_pid = ['parallelism', 'concurrency', 'err', 'err_sum', 'err_diff']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.create_opt_request = create_opt_request  # this gets updated every call

        self.use_pid_env = self.config['use_pid']
        if self.use_pid_env:
            print("[INFO] Using PID Environment")

        if self.use_pid_env:
            self.obs_cols = self.obs_cols_pid
            env_cols = ['concurrency', 'parallelism', 'read_throughput']
            self.env = ods_pid_env.PIDEnv(create_opt_req=create_opt_request, time_window="-1d",
                                          observation_columns=env_cols, target_thput=819.,
                                          action_space_discrete=False)
        else:
            self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request, time_window="-30d",
                                                    observation_columns=self.obs_cols)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = ddpg_agents.DDPGAgent(
            state_dim=state_dim, action_dim=action_dim, device=self.device,
            max_action=self.create_opt_request.max_concurrency
        )

        self.replay_buffer = memory.ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
        self.env.set_buffer(self.replay_buffer)

        self.norm_data = load_clean_norm_dataset('data/benchmark_data.csv')
        self.stats = self.norm_data.describe()

        self.pretrain_mode = self.config['pretrain']
        self.eval_mode = self.config['eval']

        if not self.eval_mode and not self.config['use_gan'] and not self.use_pid_env:
            self.warm_buffer()

        self.save_file_name = f"DDPG_{'influx_gym_env'}"
        try:
            os.mkdir('./models')
        except Exception as e:
            pass
        self.training_flag = False


        if self.eval_mode:
            self.agent.load_checkpoint(self.config['checkpoint'])
            self.agent.set_eval()
            print("[INFO] In Eval Mode")
        if self.pretrain_mode:
            print("[INFO] In Rapid Pretrain")

            if self.config['use_gan']:
                self.gen = ConvGeneratorNet(100).double().to(self.device)
                self.gen.load_state_dict(torch.load(self.config['gan_path']))
                self.gan.eval()
                print("[GAN] Generator Loaded")

        if not self.eval_mode and not self.config['use_gan'] and not self.use_pid_env:
            self.warm_buffer()

        if enable_tensorboard:
            print("[INFO] DDPG: Tensorboard Enabled")
        else:
            print("[INFO] DDPG: Tensorboard Disabled")

    def warm_buffer(self):
        print("Starting to warm the DDPG buffer")

        df = self.env.space_df
        # df = df.loc[(df != 0).any(1)]
        df = df[self.obs_cols]

        means = self.stats.loc['mean']
        stds = self.stats.loc['std']

        for i in range(df.shape[0] - 1):
            current_row = df.iloc[i]
            obs = current_row[self.obs_cols]
            action = obs[['parallelism', 'concurrency']]
            next_obs = df.iloc[i + 1]
            thrpt, rtt = env_utils.smallest_throughput_rtt(last_row=current_row)
            # reward_params = JacobReward.Params(
            #     throughput=thrpt,
            #     rtt=rtt,
            #     c=current_row['concurrency'],
            #     max_cc=self.create_opt_request.max_concurrency,
            #     p=current_row['parallelism'],
            #     max_p=self.create_opt_request.max_parallelism,
            #     # max_cpu_freq=current_row['cpu_frequency_max'],
            #     # min_cpu_freq=current_row['cpu_frequency_min'],
            #     # cpu_freq=current_row['cpu_frequency_current']
            # )
            # reward = JacobReward.calculate(reward_params)

            reward = RatioReward.construct(obs)
            norm_obs = (obs - means[self.obs_cols].to_numpy()) / \
                           (stds[self.obs_cols].to_numpy() + 1e-3)
            norm_next_obs = (next_obs - means[self.obs_cols].to_numpy()) / \
                (stds[self.obs_cols].to_numpy() + 1e-3)

            terminated = False
            if 'jobId' in current_row and 'jobId' and next_obs:
                if current_row['jobId'] != next_obs['jobId'] and next_obs['jobId'] != 0:
                    terminated = True

            self.replay_buffer.add(norm_obs, action, norm_next_obs, reward, False)

    def train(self, max_episodes=1000, batch_size=32):
        if self.eval_mode:
            return [env_utils.test_agent(self, writer, eval_episodes=15)]

        self.training_flag = True
        episode_rewards = []
        options = {'launch_job': False}
        print("[DDPG] Before the first env reset()")
        obs = self.env.reset(options=options)[0]  # gurantees job is running
        obs = np.asarray(a=obs, dtype=numpy.float64)

        action_dim = self.env.action_space.shape[0]
        dist_width = 0.1
        max_action=self.create_opt_request.max_concurrency

        means = self.stats.loc['mean']
        stds = self.stats.loc['std']
        stds = stds.where(stds > 0, other=10.)

        action_log = None
        if self.config['log_action']:
            action_log = open("actions_train.log", 'a')

        state_log = None
        if self.config['log_state']:
            state_log = open("state_train.log", 'a')


        for episode in tqdm(range(max_episodes), unit='ep'):
            episode_reward = 0
            terminated = False

            if enable_tensorboard:
                start_stamp = time.time()

            ts = 0
            while not terminated:
                if self.replay_buffer.size < 1e3:
                    action = self.env.action_space.sample()
                    # print("Sampled action space: buffer size:", self.replay_buffer.size)
                else:
                    action = (
                        self.agent.select_action(np.array(obs)) +
                        np.random.normal(0, dist_width, size=action_dim)
                    ).clip(-1, 1)
                    # print("Raw agent action", action)

                env_action = np.maximum((action + 1) * 8, 1)
                env_action = np.rint(env_action)

                if action_log:
                    action_log.write(str(env_action) + "\n")
                # action = np.clip(action, 1, 32)

                new_obs, reward, terminated, truncated, info = self.env.step(env_action, reward_type='ratio')

                if self.use_pid_env:
                    norm_obs = obs
                    norm_new_obs = new_obs
                else:
                    norm_obs = (obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)
                    norm_new_obs = (new_obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

                ts += 1
                self.replay_buffer.add(norm_obs, action, norm_new_obs, reward, terminated)
                # print("Obs: ", obs)
                # print("Action:", action)
                # print("Terminated: ", terminated)
                # print("Reward: ", reward)
                # print("Next Obs: ", new_obs)

                obs = new_obs
                if self.replay_buffer.size > batch_size and ts % 10 == 0:
                    self.agent.train(replay_buffer=self.replay_buffer, batch_size=batch_size)

                episode_reward += reward

            if terminated:
                if enable_tensorboard:
                    time_elapsed = time.time() - start_stamp
                    writer.add_scalar("Train/average_reward_step", episode_reward / ts, episode)
                    writer.add_scalar("Train/job_time", time_elapsed, episode)
                    writer.add_scalar("Train/throughput", 32. / time_elapsed, episode)

                if (episode+1) < max_episodes:
                    obs = self.env.reset(options={'launch_job': True})[0]
                    obs = np.asarray(a=obs, dtype=numpy.float64)
                    # print("Episode reward: {}", episode_reward)
                    episode_rewards.append(episode_reward)
                time.sleep(1)

            if episode % 100 == 0:
                self.agent.save_checkpoint(self.config["savefile_name"] + '_' + str(episode))
                # print("Episode ", episode, " has average reward of: ", np.mean(episode_rewards))

        self.training_flag = False
        if action_log:
            action_log.flush()
            action_log.close()
        if state_log:
            state_log.flush()
            state_log.close()

        print("[DDPG] THREAD EXITING")
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
        self.env.reset()

    def close(self):
        self.env.close()
