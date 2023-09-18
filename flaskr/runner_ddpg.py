import numpy
import torch
import os
import time

from .runner_abstract import *

from torch.utils.tensorboard import SummaryWriter
from .ods_env import env_utils
from .ods_env import ods_influx_gym_env
from .ods_env import ods_pid_env
from .ods_env.gan_env import ConvGeneratorNet
from .ods_env.ods_rewards import ArslanReward, RatioReward, JacobReward

from flaskr import classes

from .algos.ddpg import agents as ddpg_agents

from .algos.ddpg import memory
from .algos.ddpg import utils

from tqdm import tqdm

enable_tensorboard = os.getenv("ENABLE_TENSORBOARD", default='False').lower() in ('true', '1', 't')
writer = None
if enable_tensorboard:
    writer = SummaryWriter('./runs/train_pid5_ddpg/')


class DDPGTrainer(AbstractTrainer):

    def __init__(self, create_opt_request=classes.CreateOptimizerRequest, max_episodes=100, batch_size=64,
                 update_policy_time_steps=20, config_file='config/default.json'):
        super().__init__("DDPG")
        """
        self.obs_cols = ['active_core_count', 'allocatedMemory',
                         'bytes_recv', 'bytes_sent', 'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min',
                         'dropin', 'dropout', 'errin', 'errout', 'freeMemory', 'maxMemory', 'memory',
                         'packet_loss_rate', 'packets_recv',
                         'packets_sent', 'bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency',
                         'destination_latency', 'destination_rtt',
                         'jobSize', 'parallelism', 'pipelining', 'read_throughput', 'source_latency', 'source_rtt',
                         'write_throughput']
        """
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
                self.gen.eval()
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
        max_action = self.create_opt_request.max_concurrency

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

                if (episode + 1) < max_episodes:
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
