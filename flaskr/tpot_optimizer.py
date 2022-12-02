import numpy as np
import matplotlib
from .env import InfluxData
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from bayes_opt import BayesianOptimization, UtilityFunction
from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest
import pickle
import os
from tpot import TPOTRegressor




class TpotOptimizer:
    def __init__(self):

        self.bounds = {}
        self.list_name_variables = ['active_core_count', 'allocatedMemory', 'avgJobSize', 'bytes_recv',
                               'bytes_sent', 'concurrency', 'cpu_frequency_current',
                               'cpu_frequency_max', 'freeMemory', 'jobSize', 'latency', 'memory',
                               'parallelism', 'pipelining', 'rtt', 'totalBytesSent']

        self.influx = InfluxData(time_window="-60s")
        print(os.getcwd())
        with open("flaskr/tpot_model.pickle", "rb") as f:
            tpot_args = pickle.load(f)

        self.tpotModel = TPOTRegressor(template=tpot_args["template"])
        tpot_args["fitted_pipeline_"] = tpot_args["template"]
        del tpot_args["template"]
        for k, v in tpot_args.items():
            setattr(self.tpotModel, k, v)



    def create_optimizer(self, create_req: CreateOptimizerRequest):

        self.bounds['max_pipesize'] = [0, create_req.max_pipesize]
        self.bounds['max_parallel'] = [0, create_req.max_parallelism]
        self.bounds['max_concurrency'] = [0, create_req.max_concurrency]

        return self.bounds

    def delete_optimizer(delete_req: DeleteOptimizerRequest):
        print(delete_req)
        # plot_gp(delete_req.node_id)
        # del bayesian_optimizer_map[delete_req.node_id] dud
        return delete_req.node_id


# Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
# throughput, rtt]
# returns the next parameters to use

    def grid_optimizer(self,x, bounds, iters = 50):
        res = {}
        counter = 0
        print('----Optimizing----')

        for i in range(self.bounds['max_concurrency'][0], self.bounds['max_concurrency'][1]):
            for j in range(self.bounds['max_parallel'][0], self.bounds['max_parallel'][1]):
                for k in range(self.bounds['max_pipesize'][0], self.bounds['max_pipesize'][1]):
                    if counter > iters:
                        break
                    combination = (i, j, k)
                    x[5], x[12], x[13] = i,j,k
                    predicted_throughput = self.tpotModel.predict(x.values.reshape(1, len(x)))
                    res[combination] = predicted_throughput
                    counter += 1
        return sorted(res.items(), key=lambda x:x[1], reverse = True)


    def input_optimizer(self, input_req: InputOptimizerRequest):

        x = self.influx.query_bo_space(input_req.node_id)
        x = x[self.list_name_variables]
        opt_result = self.grid_optimizer(x.iloc[0], self.bounds)
        suggestion = opt_result[0][0]
        print(suggestion)
        return suggestion