import json

from flask import Flask
from flask import request
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest
from .bo_optimizer import *
from .agent import *

scheduler = BackgroundScheduler()

def at_exit():
    # scheduler.shutdown()
    agent.clean_all()

atexit.register(at_exit)

app = Flask(__name__)

@app.route('/optimizer/create', methods=['POST'])
def create_optimizer():
    if request.method == 'POST':
        json_dict = request.json
        create_opt = CreateOptimizerRequest(json_dict['nodeId'], json_dict['maxConcurrency'],
                                            json_dict['maxParallelism'], json_dict['maxPipelining'],
                                            json_dict['maxChunkSize'])
        print(create_opt.__str__())
        schedule = agent.create_optimizer(create_opt)
        if schedule:
            opt = agent.get_optimizer(create_opt.node_id)
            scheduler.add_job(opt.envs.fetch_and_train, trigger='interval', seconds=15)
            scheduler.start()
        return ('', 204)


@app.route('/optimizer/input', methods=['POST'])
def input_to_optimizer():
    if request.method == 'POST':
        jd = request.json
        input_operation = InputOptimizerRequest(jd['nodeId'], jd['throughput'], jd['rtt'], jd['concurrency'],
                                                jd['parallelism'], jd['pipelining'], jd['chunkSize'])
        print(input_operation.__str__())
        try:
            return agent.input_optimizer(input_operation), 200
        except KeyError:
            print("Failed to insert point as we have already tried this point: ")
    return '', 500


@app.route('/optimizer/delete', methods=['POST'])
def delete_optimizer():
    if request.method == 'POST':
        jd = request.json
        delete_op = DeleteOptimizerRequest(jd['nodeId'])
        print(delete_op.__str__())
        agent.delete_optimizer(delete_op)
    return '', 204
