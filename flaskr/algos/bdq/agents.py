import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from models import PreNet, AdvantageNet, StateNet
from .models import PreNet, AdvantageNet, StateNet
# from algos.abstract_agent import AbstractAgent
from ..abstract_agent import AbstractAgent
from copy import deepcopy


class BDQAgent(AbstractAgent, object):
    """
    Branching Deep-Q Agent. Inherits AbstractAgent
    """

    def __init__(self, state_dim, action_dims: list,
                 device, num_actions=1, discount=0.99, tau=0.005, decay=0.9966, evaluate=False,
                 writer=None) -> None:
        super().__init__("BDQ")
        self.device = device
        self.discount = discount
        self.tau = tau

        self.writer = writer
        self.call_count = 0

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
            self.optimizer_advs.append(optim.Adam(self.adv_nets[i].parameters(), lr=1e-3))

        self.epsilon = 1.
        # rated for 2000 episodes
        self.decay = decay

    @staticmethod
    def soft_update_agent(local: "BDQAgent", target: "BDQAgent", tau):
        """
        UP-SYNC
        """
        BDQAgent.soft_update(local.pre_net, target.pre_net, tau)
        BDQAgent.soft_update(local.state_net, target.state_net, tau)
        for i in range(local.num_actions):
            BDQAgent.soft_update(local.adv_nets[i], target.adv_nets[i], tau)

        BDQAgent.soft_update(target.pre_net, target.pre_target, target.tau)
        BDQAgent.soft_update(target.state_net, target.state_target, target.tau)
        for i in range(target.num_actions):
            BDQAgent.soft_update(target.adv_nets[i], target.adv_targets[i], target.tau)

    def update_epsilon(self):
        self.epsilon = max(0.005, self.epsilon * self.decay)

    def set_eval(self):
        self.epsilon = 0.
        self.pre_net.eval()
        self.state_net.eval()

        for adv_net in self.adv_nets:
            adv_net.eval()
        
    def select_action(self, state, bypass_epsilon=False):
        if np.random.random() <= self.epsilon and not bypass_epsilon:
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
        actions = q_values_tensor.argmax(dim=-1).detach()

        return actions.flatten().cpu().numpy()

    def compute_target_loss(self, state, next_state, actions, rewards, not_dones):
        # actual
        # print(state.shape)

        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        actions = actions.to(torch.long).transpose(0, 1)
        # act_shape = actions.shape
        actions = actions.unsqueeze(-1)

        """
        ONLINE
        """
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_nets[i](pre_state)

            selected_adv = advantages.gather(1, actions[i])
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_value + selected_adv - adv_means)

        q_values_actual = torch.stack(q_values_mat)
        q_values_actual = q_values_actual.squeeze(-1)
        
        """
        TARGET PART 1
        """
        if np.random.random() <= self.epsilon:
            # like SARSA
            acts = []
            for i in range(self.num_actions):
                act = torch.randint(0, self.action_dims[i], size=(rewards.shape[0], 1), device=self.device)
                acts.append(act)
            max_actions = torch.stack(acts)
        
        else:    
            pre_state = self.pre_net(next_state)

            state_value = self.state_net(pre_state)

            q_values_mat = []
            for i in range(self.num_actions):
                advantages = self.adv_nets[i](pre_state)

                adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
                q_values_mat.append(state_value + advantages - adv_means)

            q_values_tensor = torch.stack(q_values_mat)

            max_actions = q_values_tensor.argmax(dim=-1).unsqueeze(-1)

        """
        TARGET PART 2
        """
        pre_states = self.pre_target(next_state)

        state_values = self.state_target(pre_states)

        q_values_mat = []
        for i in range(self.num_actions):
            advantages = self.adv_targets[i](pre_states)

            selected_adv = advantages.gather(1, max_actions[i])
            adv_means = advantages.mean(dim=-1).unsqueeze(dim=-1)
            q_values_mat.append(state_values + selected_adv - adv_means)

        q_values_mat = torch.stack(q_values_mat)

        # compute sum of Q values across advantage nets
        # sum_q = self.discount * (q_values_mat.sum(0) / self.num_actions)
        # sum_q = sum_q * not_dones
        # sum_q = sum_q.transpose(0, 1)

        all_q = self.discount * (q_values_mat.squeeze(-1) / self.num_actions)
        all_q = all_q * not_dones.transpose(0, 1)
        
        rewards = rewards.transpose(0, 1)
        
        # target = rewards + sum_q
        all_target = rewards + all_q
        # target = target.detach()
        all_target = all_target.detach()

        # q_values_avg = q_values_actual.sum(0).unsqueeze(0) / self.num_actions
        
        # loss = F.mse_loss(q_values_avg, target)
        loss = F.mse_loss(q_values_actual, all_target)
        return loss, q_values_actual, all_target

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        loss, _, _ = self.compute_target_loss(state, next_state, action, reward, not_done)
        if self.writer is not None:
            self.writer.add_scalar("Train/loss", loss, self.call_count)
            self.call_count += 1

        # self.optimizer.zero_grad()
        self.optimizer_pre.zero_grad()
        self.optimizer_state.zero_grad()
        for i in range(self.num_actions):
            self.optimizer_advs[i].zero_grad()

        loss.backward()
        # self.optimizer.step()
        for i in range(self.num_actions):
            self.optimizer_advs[i].step()
        self.optimizer_state.step()
        self.optimizer_pre.step()

        BDQAgent.soft_update(self.pre_net, self.pre_target, self.tau)
        BDQAgent.soft_update(self.state_net, self.state_target, self.tau)
        for i in range(self.num_actions):
            BDQAgent.soft_update(self.adv_nets[i], self.adv_targets[i], self.tau)

    def save_checkpoint(self, filename):
        torch.save(self.pre_net.state_dict(), filename + '_pre_net')
        torch.save(self.state_net.state_dict(), filename + '_state_net')

        for i in range(self.num_actions):
            torch.save(self.adv_nets[i].state_dict(),
                       filename + '_adv_net_' + str(i))
            torch.save(self.optimizer_advs[i].state_dict(),
                       filename + '_adv_optimizer_' + str(i))

        torch.save(self.optimizer_pre.state_dict(), filename + '_pre_optimizer')
        torch.save(self.optimizer_state.state_dict(), filename + '_state_optimizer')

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
                    filename + '_adv_net_' + str(i),
                    map_location=torch.device(device)
                )
            )
            self.all_targets[adjusted_i] = deepcopy(self.all_nets[adjusted_i])

            self.optimizer_advs[i].load_state_dict(
                torch.load(
                    filename + '_adv_optimizer_' + str(i),
                    map_location=torch.device(device)
                )
            )

        self.adv_nets = self.all_nets[2:]
        self.adv_targets = self.all_nets[2:]

        # load optimizer
        # prefer to reconstruct optimizer for now because of device shenanigans
        self.module_list = nn.ModuleList(self.all_nets)
        # self.optimizer = torch.optim.Adam(self.module_list.parameters())
        self.optimizer_pre.load_state_dict(
            torch.load(
                filename + '_pre_optimizer',
                map_location=torch.device(device)
            )
        )

        self.optimizer_state.load_state_dict(
            torch.load(
                filename + '_state_optimizer',
                map_location=torch.device(device)
            )
        )
        

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
