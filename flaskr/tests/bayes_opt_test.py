import unittest

from flaskr import CreateOptimizerRequest, InfluxData, InputOptimizerRequest
from flaskr.another_bo import BayesianOptimizerOld

node_id = "onedatashare@gmail.com-mac"
max_concurrency = 20
max_parallelism = 21
max_pipesize = 1000
max_chunk_size = 1
optimizer_type = "BO"
file_count = 1


class BayesOptTest(unittest.TestCase):

    def test_create_optimizer_bayes(self):
        bayesOpt = BayesianOptimizerOld()
        bayesOpt.influx = InfluxData()
        create_req = CreateOptimizerRequest(node_id=node_id, optimizer_type=optimizer_type, max_pipesize=max_pipesize,
                                            max_concurrency=max_concurrency, max_parallelism=max_parallelism,
                                            max_chunk_size=max_chunk_size, file_count=file_count)
        bayesOpt.create_optimizer(create_req)
        self.assertTrue(bayesOpt.bayesian_optimizer_map[node_id], "Did not create bayesian Optimizer in map for node")
        self.assertTrue(bayesOpt.bo_utility_map[node_id], "Did not create utility for this node")
        self.assertTrue(bayesOpt.influx, "Did not create influx")

    def test_one_input_optimizer_bayes(self):
        bayesOpt = BayesianOptimizerOld()
        bayesOpt.influx = InfluxData(time_window="-1h")
        create_req = CreateOptimizerRequest(node_id=node_id, optimizer_type=optimizer_type, max_pipesize=max_pipesize,
                                            max_concurrency=max_concurrency, max_parallelism=max_parallelism,
                                            max_chunk_size=max_chunk_size, file_count=file_count)
        bayesOpt.create_optimizer(create_req)
        input_req = InputOptimizerRequest(node_id, 0, 0, 0, 0, 0, 0)
        res = bayesOpt.input_optimizer(input_req)
        self.assertTrue(res['concurrency'] > 0 and res['parallelism'] > 0)

    def test_duplicate_input_requests(self):
        bayesOpt = BayesianOptimizerOld()
        bayesOpt.influx = InfluxData(time_window="-1h")
        create_req = CreateOptimizerRequest(node_id=node_id, optimizer_type=optimizer_type, max_pipesize=max_pipesize,
                                            max_concurrency=max_concurrency, max_parallelism=max_parallelism,
                                            max_chunk_size=max_chunk_size, file_count=file_count)
        bayesOpt.create_optimizer(create_req)
        input_req = InputOptimizerRequest(node_id, 1, 1, 1, 1, 1, 1)
        res = bayesOpt.input_optimizer(input_req)
        print("result: ", res)
        self.assertTrue(res['concurrency'] > 0 and res['parallelism'] > 0)
        input_req.throughput=2
        res = bayesOpt.input_optimizer(input_req)
        self.assertTrue(len(res) == 2, "Input Request did not generate a tuple with 4 keys")
