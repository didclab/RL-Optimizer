import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self, state_dimension):
        super(GeneratorNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dimension, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, state_dimension),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class DiscriminatorNet(nn.Module):
    def __init__(self, state_dimension):
        super(DiscriminatorNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dimension, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class PreNet(nn.Module):
    def __init__(self, state_dimension):
        super(PreNet, self).__init__()
        self.l1 = nn.Linear(state_dimension, 512)
        self.l2 = nn.Linear(512, 256)

    def forward(self, state):
        # print(state)
        pre = state
        pre = F.relu(self.l1(pre))

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
