import numpy as np
import matplotlib
import pandas as pd
from .env import InfluxData

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest

bayesian_optimizer_map = {}
bo_utility_map = {}

list_name_variables = ['concurrency', 'parallelism', 'pipelining', 'throughput']


class BayesianOptimizerOld():
    def __init__(self):
        self.influx = InfluxData()  # every input call will select the last 10 points.
        self.bayesian_optimizer_map = {}
        self.bo_utility_map = {}

    def create_optimizer(self, create_req: CreateOptimizerRequest):
        if create_req.node_id not in self.bayesian_optimizer_map:
            pbounds = {}
            print(create_req)
            # if create_req.max_chunksize > 0:
            #     pbounds['chunkSize'] = (64000, create_req.max_chunksize)
            # if create_req.max_pipesize > 0:
            #     pbounds['pipelining'] = (1, create_req.max_pipesize)
            if create_req.max_parallelism > 1:
                pbounds['parallelism'] = (1, create_req.max_parallelism)
            if create_req.max_concurrency > 1:
                pbounds['concurrency'] = (1, create_req.file_count)
            local_opt = BayesianOptimization(
                f=self.black_box_func,
                pbounds=pbounds,
                verbose=2,
            )
            utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
            self.bayesian_optimizer_map[create_req.node_id] = local_opt
            self.bo_utility_map[create_req.node_id] = utility
        else:
            print("Optimizer already exists for", create_req.node_id)

    def input_optimizer(self, input_req: InputOptimizerRequest):
        print(input_req)
        print("BO map has: ", list(self.bayesian_optimizer_map))
        measurement_rows = self.influx.query_bo_space(input_req.node_id)
        if len(measurement_rows) < 1:
            return {
                "concurrency": input_req.concurrency,
                "parallelism": input_req.parallelism
            }
        measurement_rows.fillna(0)
        measurement_rows = measurement_rows.loc[measurement_rows["concurrency"] > 0]
        measurement_rows = measurement_rows.loc[measurement_rows["parallelism"] > 0]
        measurement_rows = measurement_rows.loc[measurement_rows["jobId"] > 0]
        print(measurement_rows)
        y = np.asarray(measurement_rows['throughput'].mean(), measurement_rows['rtt'].mean())
        x = np.array([measurement_rows['concurrency'].mean(), measurement_rows['parallelism'].mean()])
        print("Inputting points: \n x:{} \n y:{}".format(x,y))
        _uf = self.bo_utility_map[input_req.node_id]
        local_opt = self.bayesian_optimizer_map[input_req.node_id]
        try:
            local_opt.register(
                params=x,
                target=y,
            )
            print("BO has registered: {} points.".format(len(local_opt.space)), end="\n\n")
        except KeyError:
            print("BO already registered: {}".format(x), end="\n\n")

        suggestion = local_opt.suggest(_uf)
        print("Suggesting {}".format(suggestion), end="\n\n")
        # return local_opt.suggest(_uf)
        return suggestion

    def delete_optimizer(self, delete_req: DeleteOptimizerRequest):
        self.plot_gp(delete_req.node_id)
        print(delete_req)
        # del bayesian_optimizer_map[delete_req.node_id] dud
        if delete_req.node_id in self.bayesian_optimizer_map:
            self.bayesian_optimizer_map[delete_req.node_id] = None
        if delete_req.node_id in self.bo_utility_map:
            self.bo_utility_map[delete_req.node_id] = None

        return delete_req.node_id

    # Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
    # throughput, rtt]
    # returns the next parameters to use

    def black_box_func(self, throughput, rtt):
        return throughput, rtt

    def plot_gp(self, transfer_node_id):
        optimizer = self.bayesian_optimizer_map[transfer_node_id]
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
        pass

    def close(self):
        self.influx.close_client()
