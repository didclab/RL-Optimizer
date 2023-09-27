import gymnasium.spaces
import torch
import time

from .runner_abstract import *

from torch.utils.tensorboard import SummaryWriter
from .ods_env import env_utils

from .ods_env import ods_pid_env
from .ods_env.gan_env import ConvGeneratorNet
from .ods_env.ods_rewards import ArslanReward, RatioReward, JacobReward

from flaskr import classes

from .algos.bdq import agents as bdq_agents
from .algos.global_memory import ReplayBuffer as ReplayBufferBDQ

from .algos.ddpg import utils

from tqdm import tqdm

# enable_tensorboard = os.getenv("ENABLE_TENSORBOARD", default='False').lower() in ('true', '1', 't')
writer = None
enable_tensorboard = False
# if enable_tensorboard:
#     writer = SummaryWriter('./runs/debug_pid5_bdq/')


class BDQTrainer(AbstractTrainer):
    def __init__(self, create_opt_request: classes.CreateOptimizerRequest, max_episodes=100, batch_size=64,
                 config_file='config/default.json', hook=None):
        global writer, enable_tensorboard
        super().__init__("BDQ")
        print("[DEBUG/BDQTrainer]", hook, create_opt_request.host_url)

        self.config = parse_config(config_file)
        self.hook = hook  # sync function to manager

        if self.config['tensorboard_path'] is not None:
            writer = SummaryWriter(self.config['tensorboard_path'])
            enable_tensorboard = True

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
        self.host_url = create_opt_request.host_url

        self.use_pid_env = self.config['use_pid']
        if self.use_pid_env:
            print("[INFO] Using PID Environment")

        if self.use_pid_env:
            self.obs_cols = self.obs_cols_pid
            env_cols = ['concurrency', 'parallelism', 'read_throughput',
                        'cpu_frequency_current', 'cpu_frequency_max', 'cpu_frequency_min']
            self.env = ods_pid_env.PIDEnv(create_opt_req=create_opt_request, time_window="-1d",
                                          observation_columns=env_cols, target_thput=819.,
                                          action_space_discrete=True, config=self.config)
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
                self.gen.eval()
                print("[GAN] Generator Loaded")

        if not self.eval_mode and not self.config['use_gan'] and not self.use_pid_env:
            self.warm_buffer()

        if self.config['savefile_name']:
            self.save_file_name = self.config['savefile_name']
        else:
            self.save_file_name = f"BDQ_{'influx_gym_env'}"
        self.training_flag = False

        self.upsync_frequency = self.config['upsync_frequency']
        self.upsync_tau = self.config['upsync_tau']
        self.hook_frequency = self.config['hook_frequency']

        if enable_tensorboard:
            print("[INFO] Tensorboard Enabled")
        else:
            print("[INFO] Tensorboard Disabled")

    def clone_agent(self):
        """
        Needed for bootstrapping
        """
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.action_space.n

        return bdq_agents.BDQAgent(
            state_dim=state_dim, action_dims=[action_dim, action_dim], device=self.device, num_actions=2,
            decay=0.992, writer=None
        )

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
        for _ in tqdm(range(max_episodes), unit='it', total=max_episodes):
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
        # config takes priority
        if self.config['train_num_episodes'] > 0:
            max_episodes = self.config['train_num_episodes']
            print("[CONFIG] training for", str(max_episodes), "episodes")

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
        obs = np.asarray(a=obs, dtype=np.float64)
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

        cur_target = self.env.target_thput
        cur_freq = self.env.target_freq
        cur_freq_max = self.env.freq_max

        step_c = 0
        hook_counter = 0
        for episode in tqdm(range(max_episodes), unit='ep'):
            episode_reward = 0
            terminated = False

            # print("BDQTrainer.train(): starting episode", episode+1)
            eval_counter += 1
            if self.hook is not None and eval_counter % self.upsync_frequency == 0:
                bdq_agents.BDQAgent.soft_update_agent(self.agent, self.master_model, self.upsync_tau)
                hook_counter += 1

                if hook_counter == self.hook_frequency:
                    hook_counter = 0
                    self.hook()

            if eval_counter == eval_when:
                eval_counter = 0
                # 1. save model to temp
                # self.agent.save_checkpoint(temp_file_name)

                # 2. call and wait for eval
                print("\nEvaluating model at", str(episode))
                mean_reward = env_utils.test_agent(
                    self, writer, eval_episodes=eval_how_many, use_checkpoint=temp_file_name, use_id=episode
                )

                # 3. if better, save model as best
                if best_mean_rew < mean_reward:
                    best_mean_rew = mean_reward
                    self.agent.save_checkpoint("best_" + self.config['savefile_name'])

                # 4. reset environment
                obs = self.env.reset(options={'launch_job': True})[0]
                obs = np.asarray(a=obs, dtype=np.float64)

            if enable_tensorboard:
                start_stamp = time.time()
            while not terminated:
                time.sleep(5)

                if self.use_pid_env:
                    norm_obs = obs
                else:
                    norm_obs = (obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

                obs_np = np.array(norm_obs)
                # print("[Normalized Obs]", obs_np)

                if state_log:
                    state_log.write(np.array2string(obs_np, precision=3, seperator=',') + '\n')

                if self.replay_buffer.size <= 1e2: # or history_len < 4:
                    actions = [self.env.action_space.sample(), self.env.action_space.sample()]
                    # print("[Replay Buffer Size]", self.replay_buffer.size)
                else:
                    actions = self.agent.select_action(obs_np)

                params = [self.actions_to_params[a] for a in actions]
                if action_log:
                   action_log.write(str(params) + "\n")

                new_obs, reward, terminated, truncated, info = self.env.step(params, reward_type=reward_type)
                ts += 1

                if info is not None:
                    if cur_target < info['nic_speed']:
                        cur_target = info['nic_speed']
                        self.env.set_target_thput(cur_target)

                    if cur_freq > info['cpu_frequency_min'] or cur_freq_max < info['cpu_frequency_max']:
                        cur_freq = info['cpu_frequency_min']
                        cur_freq_max = info['cpu_frequency_max']
                        self.env.set_target_freq(cur_freq, cur_freq_max)

                    if enable_tensorboard:
                        writer.add_scalar("Train/cpu_freq", info['cpu_frequency_current'], step_c)
                        step_c += 1

                if self.use_pid_env:
                    norm_next_obs = new_obs
                else:
                    norm_next_obs = (new_obs - means[self.obs_cols].to_numpy()) / \
                        (stds[self.obs_cols].to_numpy() + 1e-3)

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
                if (episode+1) < max_episodes and (eval_counter < eval_when-1 or eval_when < 0):
                    obs = self.env.reset(options={'launch_job': True})[0]
                    obs = np.asarray(a=obs, dtype=np.float64)

                # print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)

                if self.replay_buffer.size > 1e2:
                    # print("[Epsilon-Decay] Updating Epsilon")
                    self.agent.update_epsilon()

                # self.agent.save_checkpoint(self.config["savefile_name"] + '_temp')

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
