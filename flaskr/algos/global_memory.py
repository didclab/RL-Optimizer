import torch
import numpy as np


class ReplayBuffer(object):

    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dimension))
        self.action = np.zeros((max_size, action_dimension))
        self.next_state = np.zeros((max_size, state_dimension))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, next_states, rewards, dones):
        batch_size = len(dones)
        first_size = min(self.max_size - self.ptr, batch_size)
        to_index = self.ptr + first_size

        self.state[self.ptr:to_index] = states
        self.action[self.ptr:to_index] = actions
        self.next_state[self.ptr:to_index] = next_states
        self.reward[self.ptr:to_index] = rewards
        self.not_done[self.ptr:to_index] = 1. - dones

        if first_size < batch_size:
            # wrap-around
            self.ptr = 0
            to_index = batch_size - first_size

            self.state[self.ptr:to_index] = states
            self.action[self.ptr:to_index] = actions
            self.next_state[self.ptr:to_index] = next_states
            self.reward[self.ptr:to_index] = rewards
            self.not_done[self.ptr:to_index] = 1. - dones

        self.size = min(self.size + batch_size, self.max_size)
        if to_index < self.max_size:
            self.ptr = to_index
        else:
            self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
