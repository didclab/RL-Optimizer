import numpy as np
import matplotlib

from . import InfluxData

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
            if create_req.max_chunksize > 0:
                pbounds['chunkSize'] = (64000, create_req.max_chunksize)
            if create_req.max_pipesize > 0:
                pbounds['pipelining'] = (1, create_req.max_pipesize)
            if create_req.max_parallelism > 1:
                pbounds['parallelism'] = (1, create_req.max_parallelism)
            if create_req.max_concurrency > 1:
                pbounds['concurrency'] = (1, create_req.max_concurrency)
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
        print(measurement_rows.columns)
        measurement_rows = measurement_rows[measurement_rows['totalBytesSent'] is not None]
        if len(measurement_rows) < 1:
            return input_req

        local_opt = self.bayesian_optimizer_map[input_req.node_id]
        y = np.asarray(measurement_rows['throughput'].mean(), measurement_rows['rtt'].mean())
        x = np.array([measurement_rows['concurrency'], measurement_rows['parallelism'], measurement_rows['pipelining']])
        _uf = self.bo_utility_map[input_req.node_id]
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
        print(delete_req)
        self.plot_gp(delete_req.node_id)
        self.bayesian_optimizer_map[delete_req.node_id] = None
        self.bo_utility_map[delete_req.node_id] = None
        # del bayesian_optimizer_map[delete_req.node_id] dud
        return delete_req.node_id

    # Inputs into optimizer the current [concurrency, parallelism, pipelining] as params, and the targets are [
    # throughput, rtt]
    # returns the next parameters to use

    def black_box_func(self, throughput, rtt):
        return throughput, rtt

    def plot_gp(self, transfer_node_id):
        optimizer = bayesian_optimizer_map[transfer_node_id]
        concurrency_list = []  # x
        parallelism_list = []  # y
        pipe_list = []  # z
        chunksize_list = []  # w
        throughput_list = []
        for key in optimizer.space._cache:
            parallelism_list.append(key[0])
            concurrency_list.append(key[1])
            pipe_list.append(key[2])
            chunksize_list.append(key[3])
            throughput_list.append(optimizer.space._cache[key])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Creating plot
        ax.scatter(concurrency_list, parallelism_list, pipe_list, c=throughput_list, cmap=plt.hot())
        ax.set_xlabel(list_name_variables[0])
        ax.set_ylabel(list_name_variables[1])
        ax.set_zlabel(list_name_variables[2])
        # plt.colorbar() # this is causing problems; missing mapping for colorbar
        plt.title('%s in function of %s, %s and %s' % (
            list_name_variables[0], list_name_variables[1], list_name_variables[2],
            list_name_variables[3]))

        plt.savefig(transfer_node_id + '.png')
        pass

    def close(self):
        self.influx.close_client()
