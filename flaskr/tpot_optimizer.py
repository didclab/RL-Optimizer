import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest
import pickle
from tpot import TPOTRegressor

with open("./tpot_model.pickle", "rb") as f:
    tpot_args = pickle.load(f)

tpotModel = TPOTRegressor(template=tpot_args["template"])
tpot_args["fitted_pipeline_"] = tpot_args["template"]
del tpot_args["template"]
for k, v in tpot_args.items():
    setattr(tpotModel, k, v)

bayesian_optimizer_map = {}
bo_utility_map = {}



list_name_variables = ['active_core_count', 'allocatedMemory', 'avgJobSize', 'bytes_recv',
       'bytes_sent', 'concurrency', 'cpu_frequency_current',
       'cpu_frequency_max', 'freeMemory', 'jobSize', 'latency', 'memory',
       'parallelism', 'pipelining', 'rtt', 'totalBytesSent']

bounds = {}

def create_optimizer(create_req: CreateOptimizerRequest):

    bounds['max_pipesize'] = [0,create_req.max_pipesize]
    bounds['max_parallel'] = [0, create_req.max_parallelism]
    bounds['max_concurrency'] = [0, create_req.max_concurrency]

    return bounds

def delete_optimizer(delete_req: DeleteOptimizerRequest):
    print(delete_req)
    # plot_gp(delete_req.node_id)
    # del bayesian_optimizer_map[delete_req.node_id] dud
    return delete_req.node_id


# Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
# throughput, rtt]
# returns the next parameters to use

def grid_optimizer(x, bounds, iters = 500):
    res = {}
    counter = 0
    for i in range(bounds['max_concurrency'][0], bounds['max_concurrency'][1]):
        for j in range(bounds['max_parallel'][0], bounds['max_parallel'][1]):
            for k in range(bounds['max_pipesize'][0], bounds['max_pipesize'][1])
                if counter>iters:
                    break
                combination = (i, j, k)
                x[5], x[12], x[13] = i,j,k
                predicted_throughput = tpotModel.predict(x.reshape(1,len(x)))
                res[combination] = predicted_throughput
                counter+=1
    return sorted(res.items(), key=lambda x:x[1])


def input_optimizer(input_req: InputOptimizerRequest, bounds):
    # local_opt = bayesian_optimizer_map[input_req.node_id]
    print(input_req)
    x = np.array()
    for var in list_name_variables:
        x.append(input_req[var])

    opt_result = grid_optimizer(x, bounds)

    suggestion = opt_result.keys()[0]

    return suggestion

