import numpy as np
import time
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest

bayesian_optimizer_map = {}
bo_utility_map = {}
bo_graph_map = {}

def create_optimizer(create_req: CreateOptimizerRequest):
    pbounds = {}
    print(create_req)
    if create_req.max_chunksize > 0:
        pbounds['chunkSize'] = (64000, create_req.max_chunksize)
    if create_req.max_pipesize > 0:
        pbounds['pipelining'] = (1, create_req.max_pipesize)
    if create_req.max_parallelism > 1:
        pbounds['parallelism'] = (1, create_req.max_parallelism)
    if create_req.max_concurrency > 1:
        pbounds['concurrency'] = (1, create_req.max_concurrency)
    local_opt = BayesianOptimization(
        f=black_box_func,
        pbounds=pbounds,
        verbose=2,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    bayesian_optimizer_map[create_req.node_id] = local_opt
    bo_utility_map[create_req.node_id] = utility


def delete_optimizer(delete_req: DeleteOptimizerRequest):
    print(delete_req)
    del bayesian_optimizer_map[delete_req.node_id]
    return delete_req.node_id


# Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
# throughput, rtt]
# returns the next parameters to use
def input_optimizer(input_req: InputOptimizerRequest):
    local_opt = bayesian_optimizer_map[input_req.node_id]
    print(input_req)
    x = np.array([input_req.concurrency, input_req.parallelism, input_req.pipelining, input_req.chunk_size])
    y = input_req.throughput
    _uf = bo_utility_map[input_req.node_id]

    local_opt.register(
        params=x,
        target=y,
    )
    print("BO has registered: {} points.".format(len(local_opt.space)), end="\n\n")
    local_opt.suggest(_uf)
    return local_opt.suggest(_uf)


def black_box_func(throughput):
    return throughput

def graph(transfer_node_id, params, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    z = np.random.standard_normal(100)
    c = np.random.standard_normal(100)

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()


