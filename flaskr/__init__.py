import json

from flask import Flask
from flask import request
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import time
from os import remove

from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest
from .bo_optimizer import *
from .agent import *

import threading
import requests
import json

scheduler = BackgroundScheduler()

with open('transfer.json') as f:
    transfer_request = json.load(f)

class ScheduleTransfer(threading.Thread):
    def __init__(self):
        super(ScheduleTransfer, self).__init__()
    
    def run(self):
        for f in args.file_path:
            remove(f)
        time.sleep(30.)
        r = requests.post(
            "http://pred.elrodrigues.com:8092/api/v1/transfer/start",
            json=transfer_request
        )
        print(r.status_code, r.reason)

schedule_thread = None

def at_exit():
    # scheduler.shutdown()
    if schedule_thread:
        schedule_thread.join()
    agent.clean_all()

atexit.register(at_exit)

app = Flask(__name__)
start = time.time()

@app.route('/optimizer/create', methods=['POST'])
def create_optimizer():
    global start
    if request.method == 'POST':
        json_dict = request.json
        create_opt = CreateOptimizerRequest(json_dict['nodeId'], json_dict['maxConcurrency'],
                                            json_dict['maxParallelism'], json_dict['maxPipelining'],
                                            json_dict['maxChunkSize'])
        print(create_opt.__str__())
        schedule = agent.create_optimizer(create_opt)
        opt = agent.get_optimizer(create_opt.node_id)
        start = time.time()
        print("Start Time:", start)
        if schedule:
            scheduler.add_job(opt.envs.fetch_and_train, trigger='interval', seconds=15)
            scheduler.start()
        else:
            print('Resetting Environment...')
            # opt.envs.reset()
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
        end = time.time()
        # agent.get_optimizer(delete_op.node_id)
        print(delete_op.__str__())
        print("^^^ JOB TIME", end - start, "SECONDS ^^^")
        with open('time.log', 'a') as f:
            f.write("JOB TIME:" + str(end - start) + " seconds\n")
        # print('Waiting for last entries...')
        # time.sleep(31.)
        agent.delete_optimizer(delete_op)
        opt = agent.get_optimizer(delete_op.node_id)
        opt.envs._done_switch = True
        schedule_thread = ScheduleTransfer().start()
    return '', 204
