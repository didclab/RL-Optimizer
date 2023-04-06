from abc import ABC, abstractmethod


class AbstractReward(ABC):
    class AbstractParams(ABC):
        def __init__(self):
            pass

    @staticmethod
    @abstractmethod
    def calculate(params: AbstractParams):
        pass


class DefaultReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, rtt, thrpt):
            super().__init__()
            self.rtt = rtt
            self.thrpt = thrpt

    @staticmethod
    def calculate(params: Params):
        rtt = params.rtt
        thrpt = params.thrpt

        return rtt * thrpt


class ArslanReward(AbstractReward):
    class Params(AbstractReward.AbstractParams):
        def __init__(self, aggregate_plr, throughput, past_utility, k=1, b=0.1,
                     pos_rew=1., neg_rew=-1., pos_thresh=0.2, neg_thresh=-0.2):
            super().__init__()
            self.ag_plr = aggregate_plr
            self.throughput = throughput
            self.past_u = past_utility
            self.K = k
            self.B = b

            self.pos_rew = pos_rew
            self.pos_thresh = pos_thresh
            self.neg_rew = neg_rew
            self.neg_thresh = neg_thresh

    @staticmethod
    def calculate(params: Params):
        # compute current utility
        utility = params.throughput / params.K
        utility -= (params.B * params.throughput * params.ag_plr)

        diff = utility - params.past_u
        reward = 0.
        if diff > params.pos_thresh:
            reward = params.pos_rew
        elif diff < params.neg_thresh:
            reward = params.neg_rew

        return reward, utility

