import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    def __init__(self, state_dimension):
        super(PreNet, self).__init__()
        self.l1 = nn.Linear(state_dimension, 512)
        self.l2 = nn.Linear(512, 256)

    def forward(self, state):
        pre = F.relu(self.l1(state))
        return self.l2(pre)


class StateNet(nn.Module):
    def __init__(self, state_dimension):
        super(StateNet, self).__init__()
        self.l1 = nn.Linear(state_dimension, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        return self.l2(v)


class AdvantageNet(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super(AdvantageNet, self).__init__()
        self.l1 = nn.Linear(state_dimension, 128)
        self.l2 = nn.Linear(128, action_dimension)

    def forward(self, state):
        a = F.relu(self.l1(state))
        return self.l2(a)
