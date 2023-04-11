import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    def __init__(self, state_dimension, eval=False, normalize=False):
        super(PreNet, self).__init__()
        self.b0 = nn.BatchNorm1d(state_dimension)
        self.l1 = nn.Linear(state_dimension, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.l2 = nn.Linear(512, 256)

        self.norm = normalize

        if eval:
            self.b0.eval()
            self.b1.eval()

    def forward(self, state):
        # print(state)
        pre = state
        if self.norm:
            pre = self.b0(state)
        pre = F.relu(self.l1(pre))
        if self.norm:
            pre = self.b1(pre)
        return self.l2(pre)

    def forward_skip(self, state):
        pre = F.relu(self.l1(state))
        return self.l2(pre)


class StateNet(nn.Module):
    def __init__(self):
        super(StateNet, self).__init__()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        return self.l2(v)


class AdvantageNet(nn.Module):
    def __init__(self, action_dimension):
        super(AdvantageNet, self).__init__()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, action_dimension)

    def forward(self, state):
        a = F.relu(self.l1(state))
        return self.l2(a)
