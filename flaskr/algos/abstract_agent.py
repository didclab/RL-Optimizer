from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    Agent template. Inherit if needed
    """

    def __init__(self):
        pass

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    @abstractmethod
    def load_checkpoint(self, filename):
        pass

    @abstractmethod
    def save_checkpoint(self, filename):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train(self):
        pass
