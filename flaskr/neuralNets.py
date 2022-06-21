import torch
import numpy as np
import gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import time

from config import *

class ActorNet(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150, dtype=torch.float64)
        # self.fc2 = nn.Linear(150, 128, dtype=torch.float64)
        self.fc3 = nn.Linear(150, output_dim, dtype=torch.float64)

  # x is a tensor
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class CriticNet(nn.Module):
    def __init__(self, input_dim=8, output_dim=1):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150, dtype=torch.float64)
        self.fc2 = nn.Linear(150, 128, dtype=torch.float64)
        self.fc3 = nn.Linear(128, output_dim, dtype=torch.float64)

  # x is a tensor
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
