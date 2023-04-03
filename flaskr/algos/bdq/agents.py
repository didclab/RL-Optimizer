import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import PreNet, AdvantageNet, StateNet
from copy import deepcopy


class BDQAgent(object):
    """
    @staticmethod
    @property
    """

    def __init__(self, state_dim, action_dims, device, discount=0.99, tau=0.005) -> None:
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

        self.all_nets.append(PreNet(state_dim).to(device))
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
        self.pre_optimizer = torch.optim.Adam(self.module_list.parameters())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        pre_state = self.pre_net(state)

        state_value = self.state_net(pre_state)
        q_values = []
        for i in range(self.num_actions):
            q_values.append(state_value + self.adv_nets[i](state_value))
