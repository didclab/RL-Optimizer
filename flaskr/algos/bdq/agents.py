import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models import PreNet, AdvantageNet, StateNet
# from ..abstract_agent import AbstractAgent
from algos.abstract_agent import AbstractAgent
from copy import deepcopy


class BDQAgent(AbstractAgent, object):
    """
    Branching Deep-Q Agent. Inherits AbstractAgent
    """

    def __init__(self, state_dim, action_dims: list,
                 device, num_actions=1, discount=0.99, tau=0.005, evaluate=False) -> None:
        super().__init__()
        self.device = device
        self.discount = discount
        self.tau = tau

        """
        Index 0: PreNet Online
        Index 1: StateNet Online
        Index [2, ...]: AdvNet Online
        """
        self.all_nets = []
        self.all_targets = []

        self.num_actions = num_actions
        self.action_dims = action_dims

        self.all_nets.append(PreNet(state_dim, eval=evaluate).to(device))
        self.all_targets.append(deepcopy(self.all_nets[0]))
        self.pre_net = self.all_nets[0]
        self.pre_target = self.all_targets[0]

        self.all_nets.append(StateNet().to(device))
        self.all_targets.append(deepcopy(self.all_nets[1]))
        self.state_net = self.all_nets[1]
        self.state_target = self.all_targets[1]

        for i in range(self.num_actions):
            self.all_nets.append(AdvantageNet(action_dims[i]).to(device))
            self.all_targets.append(deepcopy(self.all_nets[-1]))
        self.adv_nets = self.all_nets[2:]
        self.adv_targets = self.all_targets[2:]
        assert len(self.adv_nets) == self.num_actions, \
            len(self.adv_targets) == self.num_actions

        self.module_list = nn.ModuleList(self.all_nets)
        self.optimizer = optim.Adam(self.module_list.parameters())

        self.optimizer_pre = optim.Adam(self.pre_net.parameters(), lr=1e-3)
        self.optimizer_state = optim.Adam(self.state_net.parameters(), lr=1e-3)
        self.optimizer_advs = []
        for i in range(self.num_actions):
            self.optimizer_advs.append(optim.Adam(
                self.adv_nets[i].parameters(), lr=1e-3))

        self.epsilon = 1.
        # rated for 2000 episodes
        self.decay = 0.9966

    def update_epsilon(self):
        self.epsilon = max(0.001, self.epsilon * self.decay)

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            # needs to be improved but will do for now
            return np.array([np.random.randint(0, self.action_dims[i]) for i in range(self.num_actions)])

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)

        q_values = []
        for i in range(self.num_actions):
            advantages = self.adv_nets[i](pre_state)
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values.append(state_value + advantages - adv_means)

        q_values_tensor = torch.stack(q_values)
        actions = q_values_tensor.argmax(dim=-1)

        return actions.flatten().cpu().numpy()

    def compute_target_loss(self, state, rewards):
        # actual
        # print(state.shape)

        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = state.to(self.device)
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_nets[i](pre_state)

            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_value + advantages - adv_means)

        q_values_tensor = torch.stack(q_values_mat)

        # max_actions = q_values_tensor.argmax(dim=-1)
        q_values_actual, test_index = q_values_tensor.max(dim=-1)
        # print(q_values_tensor.shape)
        # print(q_values_actual.shape)

        # target
        pre_states = self.pre_target(state)

        state_values = self.state_target(pre_states)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_targets[i](pre_states)
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_values + advantages - adv_means)

        q_values_mat = torch.stack(q_values_mat)
        # print(test_index[:, 0])
        # print(test_index[1])
        # print(q_values_mat.shape)
        # a = 1/0

        # scan rows of max actions
        num_samples = test_index.shape[1]
        trajectory = []

        for j in range(num_samples):
            sum_q = 0
            for i in range(self.num_actions):
                action = test_index[i, j]
                q_values = q_values_mat[i, j, action]

                sum_q += q_values

            sum_q = sum_q / self.num_actions
            trajectory.append(self.discount * sum_q)

        trajectory = torch.stack(trajectory).reshape(1, num_samples)
        rewards = rewards.transpose(0, 1)
        target = rewards + trajectory
        target = target.detach()

        loss = F.mse_loss(q_values_actual, target)
        return loss, q_values_actual, target

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        loss, q_values_actual, target = self.compute_target_loss(state, reward)

        # self.optimizer.zero_grad()
        self.optimizer_pre.zero_grad()
        self.optimizer_state.zero_grad()
        for i in range(self.num_actions):
            self.optimizer_advs[i].zero_grad()

        loss.backward()
        # self.optimizer.step()
        self.optimizer_pre.step()
        self.optimizer_state.step()
        for i in range(self.num_actions):
            self.optimizer_advs[i].step()

        BDQAgent.soft_update(self.pre_net, self.pre_target, self.tau)
        BDQAgent.soft_update(self.state_net, self.state_target, self.tau)
        for i in range(self.num_actions):
            BDQAgent.soft_update(
                self.adv_targets[i], self.adv_nets[i], self.tau)

    def save_checkpoint(self, filename):
        torch.save(self.pre_net.state_dict(), filename + '_pre_net')
        torch.save(self.state_net.state_dict(), filename + '_state_net')

        for i in range(self.num_actions):
            torch.save(self.adv_nets[i].state_dict(),
                       filename + '_adv_net_' + i)

        torch.save(self.optimizer.state_dict(), filename + '_bdq_optimizer')

    def load_checkpoint(self, filename):
        device = self.device

        self.all_nets[0].load_state_dict(
            torch.load(
                filename + '_pre_net',
                map_location=torch.device(device)
            )
        )
        # shouldn't have to move model to device with to() if same device
        self.all_targets[0] = deepcopy(self.all_nets[0])
        self.pre_net = self.all_nets[0]
        self.pre_target = self.all_targets[0]

        self.all_nets[1].load_state_dict(
            torch.load(
                filename + '_state_net',
                map_location=torch.device(device)
            )
        )
        self.all_targets[1] = deepcopy(self.all_nets[1])
        self.state_net = self.all_nets[1]
        self.state_target = self.all_targets[1]

        for i in range(self.num_actions):
            adjusted_i = i + 2
            self.all_nets[adjusted_i].load_state_dict(
                torch.load(
                    filename + '_adv_net_' + i,
                    map_location=torch.device(device)
                )
            )
            self.all_targets[adjusted_i] = deepcopy(self.all_nets[adjusted_i])

        self.adv_nets = self.all_nets[2:]
        self.adv_targets = self.all_nets[2:]

        # load optimizer
        # prefer to reconstruct optimizer for now because of device shenanigans
        self.module_list = nn.ModuleList(self.all_nets)
        self.optimizer = torch.optim.Adam(self.module_list.parameters())

    def get_tau(self):
        return self.tau

    def get_pre_net(self):
        return self.pre_net

    def get_pre_target(self):
        return self.pre_target

    def get_state_net(self):
        return self.state_net

    def get_state_target(self):
        return self.state_target

    def get_adv_nets(self):
        return self.adv_nets

    def get_adv_target(self):
        return self.adv_targets
