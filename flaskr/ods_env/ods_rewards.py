import math
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
        def __init__(self, throughput, rtt, max_cpu_freq, min_cpu_freq,cpu_freq, c, p, max_cc, max_p):
            # super().__init__()
            self.throughput = throughput
            self.rtt = rtt
            self.hyper_rtt = .1
            self.cc = c
            self.max_cc = max_cc
            self.hyper_cc = .01
            self.p = p
            self.hyper_p = .01
            self.max_p = max_p
            self.cpu_freq = cpu_freq
            self.hyper_cpu_freq = .001
            self.min_cpu_freq = min_cpu_freq
            self.max_cpu_freq = max_cpu_freq

    @staticmethod
    def calculate(params: Params):
        scaled_p = params.p / params.max_p
        scaled_p = scaled_p * params.hyper_p

        scaled_cc = params.cc / params.max_cc
        scaled_cc = scaled_cc * params.hyper_cc

        scale_cpu_freq = (params.cpu_freq - params.min_cpu_freq) / (params.max_cpu_freq - params.min_cpu_freq)
        scale_cpu_freq = params.hyper_cpu_freq * scale_cpu_freq
        rtt_disc = 1/(1 + params.hyper_rtt * math.log(params.rtt))
        reward = params.throughput * (rtt_disc * scale_cpu_freq * scaled_cc * scaled_p)
        return reward


class ArslanReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, penalty, throughput, past_utility, concurrency, parallelism, K=1.0072, b=0.02,
                     pos_rew=1., neg_rew=-1., pos_thresh=100, neg_thresh=-100):
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
        utility = params.throughput / np.power(params.K, params.total_threads)
        utility -= (params.B * params.throughput * params.penalty)

        diff = utility - params.past_u
        reward = 0.
        if diff > params.pos_thresh:
            reward = params.pos_rew
        elif diff < params.neg_thresh:
            reward = params.neg_rew

        return reward, utility
