from abc import ABC, abstractmethod

import numpy as np


class AbstractReward(ABC):
    class AbstractParams(ABC):
        @abstractmethod
        def __init__(self, *args, **kwargs):
            pass

    @staticmethod
    def construct(x):
        return x

    @staticmethod
    @abstractmethod
    def calculate(params: AbstractParams):
        pass


class DefaultReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, rtt, throughput):
            # super().__init__()
            self.rtt = rtt
            self.throughput = throughput

    @staticmethod
    def calculate(params: Params):
        rtt = params.rtt
        thrpt = params.throughput

        return rtt * thrpt


class JacobReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, throughput, rtt, total_bytes, last_concurrency, last_parallelism, action_space_max):
            # super().__init__()
            self.throughput = throughput
            self.rtt = rtt
            self.total_bytes = total_bytes
            self.action_space_max = action_space_max
            # why add and not multiply?
            self.last_action = last_parallelism + last_concurrency

    @staticmethod
    def calculate(params: Params):
        byte_ratio = ((params.throughput / 8) * params.rtt) / params.total_bytes
        action_ratio = params.last_action / params.action_space_max

        print("Byte Ratio=", byte_ratio,
              " thrpt=", params.throughput, " * rtt=", params.rtt, " /totalBytes", params.total_bytes)
        print("Action Ratio=", action_ratio, "last_action=", params.last_action, "/ action space max=",
              params.action_space_max)

        return byte_ratio / action_ratio


class ArslanReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, penalty, throughput, past_utility, concurrency, parallelism, K=1.0072, b=0.02,
                     pos_rew=1., neg_rew=-1., pos_thresh=100, neg_thresh=-100, bwidth=1.):
            # super().__init__()
            self.penalty = penalty
            self.throughput = throughput
            self.past_u = past_utility
            self.K = K
            self.B = b

            self.pos_rew = pos_rew
            self.pos_thresh = pos_thresh
            self.neg_rew = neg_rew
            self.neg_thresh = neg_thresh

            self.s = 1/bwidth

            self.total_threads = parallelism * concurrency

    @staticmethod
    def construct(x, penalty='diff_dropin', K=1.0072, b=0.02):
        t = np.minimum(x.read_throughput, x.write_throughput)
        p = x.parallelism.to_numpy()
        cc = x.concurrency.to_numpy()

        pen = x[penalty].to_numpy()

        return (t / np.power(K, p * cc)) - (b * t * pen)

    @staticmethod
    def compare(past_utility, utility, pos_rew=1., neg_rew=-1., pos_thresh=100, neg_thresh=-100):
        diff = utility - past_utility
        reward = 0.
        if diff > pos_thresh:
            reward = pos_rew
        elif diff < neg_thresh:
            reward = neg_rew
        return reward

    @staticmethod
    def calculate(params: Params):
        # compute current utility
        utility = (params.s * params.throughput) / np.power(params.K, params.total_threads)
        utility -= (params.B * params.throughput * params.penalty * params.s)

        diff = utility - params.past_u
        reward = 0.
        if diff > params.pos_thresh:
            reward = params.pos_rew
        elif diff < params.neg_thresh:
            reward = params.neg_rew

        return reward, utility
