from replayMemory import *
from neuralNets import *
from config import *


def run_worker(N, g_net, critic_net, env, device, gamma, max_episodes):
    history = []
    memory = ReplaySlice(N)
    thresh = -20 if N == 0 else -15
    l_net = g_net
    opt = optim.Adam(l_net.parameters(), lr=LEARN_RATE_A)
    time_epsilon = 1
    time_ep_decay = 0.99

    print('Worker', N, ': Started')
    for i in range(max_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        I = 1.

        while not done:
            # action, log_p = worker_step(obs)
            # Take action with probability from policy
            observation = torch.tensor(np.array([obs]), device=device, dtype=torch.float64)
            # actions is softmax'd
            # actions = l_net(observation)
            actions = l_net(observation)
            # print(N, 'wangy3')
            observation.detach()
            # print(actions)
            # https://pytorch.org/docs/stable/distributions.html#categorical
            cats = Categorical(actions)
            chosen_action = cats.sample()
            # https://pytorch.org/docs/stable/distributions.html#score-function
            # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
            action = chosen_action.item()
            log_p = cats.log_prob(chosen_action)

            obs_unsq = torch.tensor(np.array([obs]), device=device, dtype=torch.float64)
            next_obs, reward, done, _ = env.step(action)
            next_obs_list = next_obs
            next_obs = torch.tensor(np.array([next_obs]), device=device, dtype=torch.float64)
            total_reward += reward

            stval = critic_net(obs_unsq).double()
            next_stval = torch.tensor(np.array([[reward]]), device=device).double()
            if not done:
                next_stval += (gamma * critic_net(next_obs)).double()

            # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
            advantage = next_stval.item() - stval.item()
            memory.add(obs, action, reward, next_obs_list, done, log_p, I, advantage)

            actor_loss = -log_p * advantage
            actor_loss *= I

            # Backpropagate Actor
            # Black magic here
            opt.zero_grad()
            actor_loss.backward()
            # for lp, gp in zip(l_net.parameters(), g_net.parameters()):
            #     gp._grad = lp.grad
            opt.step()

            obs = next_obs_list
            I *= gamma

        if time_epsilon > 0.1:
            time.sleep(time_epsilon)
            time_epsilon *= time_ep_decay

        # self.epsilon = max(self.epsilon * self.ep_decay, 0.1)
        history.append(total_reward)
        # self.episodes.append(self.counter)
        # self.counter += 1
        mean_time = np.mean(history[thresh:])
        # self.total_reward_ep.append(total_reward)
        if i % 10 == 0:
            print('Worker', N, ': At episode', i, ': mean score of', mean_time)
        if mean_time > TH_SCORE:
            print('Worker', N, ': Optimal Agent Achieved')
            # status.status[N] = True
            memory.free()
            torch.save(l_net.state_dict(), './assign3_actor_impala_cartp_' + str(N) + '.pth')
            torch.save(critic_net.state_dict(), './assign3_critic_impala_cartp_' + str(N) + '.pth')
            return
    print('Worker', N, ': Finished Training')
    # status.status[N] = True
    memory.free()
    torch.save(critic_net.state_dict(), './assign3_critic_impala_cartp_' + str(N) + '.pth')
    return


class ActorCritic_Hogwild:
    def __init__(self, env) -> None:
        self.device = device
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_actions = env.action_space.n
        self.max_episodes = 3000  # const
        self.alpha_A = LEARN_RATE_A  # const
        self.alpha_C = LEARN_RATE_C  # const
        self.gamma = 0.99  # discount factor
        self.total_reward = []
        self.total_reward_ep = []
        self.episodes = []
        self.Actor_func = ActorNet().to(self.device)
        self.Critic_func = CriticNet().to(self.device)
        self.Critic_target = CriticNet().to(self.device)
        self.Critic_target.load_state_dict(self.Critic_func.state_dict())

        # self.Q_func = QNet().to(self.device)
        self.batch_size = MINIBAT_SIZE
        self.criterion = nn.MSELoss()
        # self.optimizer_A = SharedAdam(self.Actor_func.parameters(), lr=LEARN_RATE_A) # SGD
        self.optimizer_C = optim.Adam(self.Critic_func.parameters(), lr=LEARN_RATE_C)  # SGD
        # self.optimizer_Q = optim.Adam(self.Q_func.parameters(), lr=LEARN_RATE_C) # SGD
        self.counter = 0
        self.rng = np.random.default_rng()
        self.memory = ReplayMemory()

        # self.worker_slices = [ReplaySlice(self.memory, i*REPLAYMEM_WORKER_SIZE, (i+1)*REPLAYMEM_WORKER_SIZE) for i in range(N_WORKERS)]
        self.worker_context = None

        self.C = 1  # target update frequency
        self.epsilon = 1
        self.ep_decay = 0.9

    def reset(self):
        self.Actor_func = ActorNet().to(self.device)
        self.Critic_func = CriticNet().to(self.device)
        self.Target_func.load_state_dict(self.Policy_func.state_dict())
        self.epsilon = 1
        self.total_reward = []
        self.counter = 0

    def step(self, observation):
        # Take action with probability from policy
        observation = torch.tensor(np.array([observation]), device=self.device, dtype=torch.float64)
        # actions is softmax'd
        actions = self.Actor_func(observation)
        observation.detach()
        # print(actions)
        # https://pytorch.org/docs/stable/distributions.html#categorical
        cats = Categorical(actions)
        chosen_action = cats.sample()
        # https://pytorch.org/docs/stable/distributions.html#score-function
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return chosen_action.item(), cats.log_prob(chosen_action)

    def warm_memory(self):
        obs = self.env.reset()
        # obs = np.reshape(obs, [1, 4])
        done = False
        I = 1.
        for _ in range(REPLAYMEM_SIZE):
            action, logp = self.step(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs_list = next_obs
            obs_unsq = torch.tensor(np.array([obs]), device=self.device, dtype=torch.float64)
            next_obs = torch.tensor(np.array([next_obs]), device=self.device, dtype=torch.float64)

            stval = self.Critic_func(obs_unsq).double()
            next_stval = torch.tensor(np.array([[reward]]), device=self.device).double()
            if not done:
                next_stval += (self.gamma * self.Critic_func(next_obs)).double()

            advantage = next_stval.item() - stval.item()
            self.memory.g_add(obs, action, reward, next_obs_list, done, logp, I, advantage)
            if done:
                obs = self.env.reset()
                I = 1.
                done = False
            else:
                obs = next_obs_list
                I *= self.gamma
        print("Warmed Up ...")

    def envstep(self, observation):
        # take greedy action
        observation = torch.tensor(np.array([observation]), device=device, dtype=torch.float32)
        with torch.no_grad():
            # action = self.Policy_func(observation).max(1)[1].item()
            actions = self.Actor_func(observation).detach()
            actions = actions.unsqueeze(0)
            return torch.argmax(actions).item()

    def replay(self, t):
        if self.memory.mem_count < MINIBAT_SIZE:
            return

        obs_samples, actions, rewards, next_obs_samples, not_dones, logp, I = self.memory.random_sample()
        # rnum = self.rng.random()
        # if t % 10 == 0: # take priority sample
        #     obs_samples, actions, rewards, next_obs_samples, not_dones, logp, I = self.memory.priority_sample()
        # else:
        #     obs_samples, actions, rewards, next_obs_samples, not_dones, logp, I = self.memory.random_sample()
        target = 0  # target

        obs_samples = torch.tensor(obs_samples, device=device, dtype=torch.float64)
        actions = torch.tensor(actions, device=device, dtype=torch.int64)
        next_obs_samples = torch.tensor(next_obs_samples, device=device, dtype=torch.float64)
        rewards = torch.tensor(rewards, device=device)
        not_dones = torch.tensor(not_dones, device=device)
        logp = torch.tensor(logp, device=device, dtype=torch.float64)
        I = torch.tensor(I, device=device, dtype=torch.float64)

        # predictions
        s_values = self.Critic_func(obs_samples).squeeze()

        next_s_values = self.Critic_func(next_obs_samples).squeeze()

        expected_s_values = rewards + (self.gamma * next_s_values * not_dones)

        self.optimizer_C.zero_grad()
        loss = self.criterion(s_values, expected_s_values)
        loss.backward()
        self.optimizer_C.step()


if __name__ == "__main__":
    env3 = gym.make('LunarLander-v2')
    agent3 = ActorCritic_Hogwild(env3)
    agent3.Critic_target.share_memory()
    agent3.Actor_func.share_memory()
    agent3.warm_memory()
    worker_context = mp.spawn(run_worker,
                              args=(agent3.Actor_func,  # agent3.optimizer_A,
                                    agent3.Critic_target, gym.make('LunarLander-v2'), agent3.device,
                                    agent3.gamma, agent3.max_episodes),
                              nprocs=N_WORKERS,
                              join=False)
    t = 1
    while not worker_context.join(timeout=0.2):
        agent3.replay(t)
        t += 1
        if t % 25 == 0:
            agent3.Critic_target.load_state_dict(agent3.Critic_func.state_dict())
        if t % 50 == 0:
            print('Main Thread:', t)

    agent3.memory.free()
    torch.save(agent3.Actor_func.state_dict(), './assign3_actor_impala_cartp_main.pth')
    torch.save(agent3.Critic_func.state_dict(), './assign3_critic_impala_cartp_main.pth')
    print('Main Thread: Finished Training')
