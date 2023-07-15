import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


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
        def __init__(self, throughput, rtt, c, p, max_cc, max_p):
            # super().__init__()
            self.throughput = throughput
            self.rtt = rtt
            self.hyper_rtt = 1
            self.cc = c
            self.max_cc = max_cc
            self.hyper_cc = 1
            self.p = p
            self.hyper_p = 1
            self.max_p = max_p
            self.hyper_cpu_freq = .1

    @staticmethod
    def calculate(params: Params):
        # print("Params: ", vars(params))
        reward = params.throughput / 10000 #normalizing for a 10Gbps link
        norm_thrpt = 2* (reward-.5) #normalizing between (-1,-1)
        # print("Throughput: ", reward, "Mbps")
        # pen_rtt = 1-(params.rtt/(1000 * params.hyper_rtt))
        # reward = reward * pen_rtt
        # print("Discounting RTT: ", reward)
        # reward = reward / params.cc
        # print("Discounting Conc: ", reward)
        # reward = reward / params.p
        # print("Discounting P: ", reward)
        return norm_thrpt


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

            self.s = 1 / bwidth

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


class RatioReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, read_throughput, write_throughput, read_over_write=True):
            self.read_throughput = read_throughput
            self.write_throughput = write_throughput
            self.r_w = read_over_write

    @staticmethod
    def construct(x, r_w=True):
        if r_w:
            return x.read_throughput / (x.write_throughput + 1e-5)
        else:
            return x.write_throughput / (x.read_throughput + 1e-5)

    @staticmethod
    def calculate(params: Params):
        if params.r_w:
            reward = params.read_throughput.to_numpy() / (params.write_throughput.to_numpy() + 1e-5)
            return reward[0]
        else:
            reward = params.write_throughput.to_numpy() / (params.read_throughput.to_numpy() + 1e-5)
            return reward[0]
