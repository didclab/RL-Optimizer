import numpy as np
from bayes_opt import BayesianOptimization
from Classes import CreateOptimizerRequest
from Classes import DeleteOptimizerRequest
from Classes import InputOptimizerRequest

bayesian_optimizer_map = {}


def create_optimizer(create_req: CreateOptimizerRequest):
    pbounds = {'concurrency': (1, create_req.max_concurrency), 'parallelism': (1, create_req.max_parallelism),
               'pipelining': (1, create_req.max_pipesize), 'chunkSize': (64000, create_req.max_chunksize)}
    local_opt = BayesianOptimization(
        f=black_box_func,
        pbounds=pbounds
    )
    bayesian_optimizer_map[create_req.node_id] = local_opt
    return ''


def delete_optimizer(delete_req: DeleteOptimizerRequest):
    del bayesian_optimizer_map[delete_req.node_id]
    return delete_req.node_id


# Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
# throughput, rtt]
# returns the next parameters to use
def input_optimizer(input_req: InputOptimizerRequest):
    local_opt = bayesian_optimizer_map[input_req.node_id]
    x = np.asarray([input_req.concurrency, input_req.parallelism, input_req.pipelining, input_req.chunk_size])
    y = np.asarray([input_req.throughput, input_req.rtt])
    try:
        local_opt.register(
            params=x,
            target=y,
        )
        print("BO has registered: {} points.".format(len(local_opt.space)), end="\n\n")
    except KeyError:
        pass
    finally:
        return local_opt.suggest()


def black_box_func(throughput, rtt):
    return throughput, rtt
