import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import PreNet, AdvantageNet, StateNet
from ..abstract_agent import AbstractAgent
from copy import deepcopy


class BDQAgent(AbstractAgent, object):
    """
    Branching Deep-Q Agent. Inherits AbstractAgent
    """

    def __init__(self, state_dim, action_dims, device, discount=0.99, tau=0.005, evaluate=False) -> None:
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
        self.num_actions = len(action_dims)

        self.all_nets.append(PreNet(state_dim, eval=evaluate).to(device))
        self.all_targets.append(deepcopy(self.all_nets[0]))
        self.pre_net = self.all_nets[0]
        self.pre_target = self.all_targets[0]

        self.all_nets.append(StateNet(state_dim).to(device))
        self.all_targets.append(deepcopy(self.all_nets[1]))
        self.state_net = self.all_nets[1]
        self.state_target = self.all_targets[1]

        for i in range(self.num_actions):
            self.all_nets.append(AdvantageNet(
                state_dim, action_dims[i]).to(device))
            self.all_targets.append(deepcopy(self.all_nets[-1]))
        self.adv_nets = self.all_nets[2:]
        self.adv_targets = self.all_targets[2:]
        assert len(self.adv_nets) == self.num_actions, \
            len(self.adv_targets) == self.num_actions

        self.module_list = nn.ModuleList(self.all_nets)
        self.optimizer = torch.optim.Adam(self.module_list.parameters())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)

        q_values = []
        for i in range(self.num_actions):
            advantages = self.adv_nets[i](state_value)
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values.append(state_value + advantages - adv_means)

        q_values_tensor = torch.stack(q_values)
        actions = q_values_tensor.argmax(dim=-1)
        return actions.cpu().numpy()

    def compute_target_loss(self, state, rewards):
        # actual
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_nets[i](state_value)
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_value + advantages - adv_means)

        q_values_tensor = torch.stack(q_values_mat)
        max_actions = q_values_tensor.argmax(dim=-1)
        q_values_actual = q_values_tensor.max(dim=-1)

        # target
        pre_states = self.pre_target(state)

        state_values = self.state_target(pre_states)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_targets[i](state_values)
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_values + advantages - adv_means)

        # scan rows of max actions
        num_samples = max_actions.shape[0]
        trajectory = []
        for i in range(num_samples):
            actions = max_actions[i]
            q_values = q_values_mat[i]

            sum_q = 0
            for j in range(self.num_actions):
                sum_q += q_values[int(actions[j])]

            sum_q = sum_q / self.num_actions
            trajectory.append(self.discount * sum_q)

        trajectory = torch.stack(trajectory)
        target = rewards + trajectory
        target = target.detach()

        loss = F.mse_loss(q_values_actual, target)
        return loss, q_values_actual, target

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        loss, q_values_actual, target = self.compute_target_loss(state, reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
