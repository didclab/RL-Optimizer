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

import random

scheduler = BackgroundScheduler()

with open('transfer.json') as f:
    transfer_requests = [json.load(f)]

with open('slow-transfer.json') as f:
    transfer_requests.append(json.load(f))

transfer_request = transfer_requests[0]

start_p = 2
start_c = 2
epsilon = 1.
epsilon_decay = 0.93325
sample_space = [i for i in range(2, 7)]

num_episodes = 0
fast_slow_switch = 0
log_counts = [0, 0]


class ScheduleTransfer(threading.Thread):
    def __init__(self):
        super(ScheduleTransfer, self).__init__()

    def run(self):
        for f in args.file_path:
            try:
                remove(f)
            except:
                pass
        time.sleep(30.)

        # Inject Parameters
        transfer_request['options']['concurrencyThreadCount'] = start_c
        transfer_request['options']['parallelThreadCount'] = start_p

        r = requests.post(
            "http://192.5.87.31:8092/api/v1/transfer/start",
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
        # else:
        #     print('Resetting Environment...')
        #     opt.envs.reset()
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
        global epsilon
        global start_p
        global start_c
        global num_episodes
        global schedule_thread
        global transfer_request
        global fast_slow_switch
        global log_counts
        global scheduler

        jd = request.json
        delete_op = DeleteOptimizerRequest(jd['nodeId'])
        end = time.time()
        # agent.get_optimizer(delete_op.node_id)
        print(delete_op.__str__())
        thrput = 256 / (end - start)
        print("^^^ JOB TIME", end - start, "SECONDS ^^^")
        print("THROUGHPUT:", thrput)
        with open('time.log', 'a') as f:
            f.write(str(end - start) + "," + str(thrput) + "\n")
        # print('Waiting for last entries...')
        # time.sleep(31.)
        agent.delete_optimizer(delete_op)
        opt = agent.get_optimizer(delete_op.node_id)
        opt.envs._done_switch = True
        print('Updating distribution...')
        opt.envs.parameter_dist_map.update_parameter_dist(
            opt.envs.best_start[0],
            opt.envs.best_start[1],
            (thrput / 16) * 15
        )
        print("Resetting environment...", epsilon)

        golden_number = random.random()
        if golden_number < epsilon:
            sampled_p = random.choice(sample_space)
            sampled_c = random.choice(sample_space)
            opt.envs.set_best_action(sampled_p, sampled_c)
        else:
            b_a = opt.envs.parameter_dist_map.get_best_parameter()
            opt.envs.set_best_action(b_a[0], b_a[1])

        opt.reset_obs = opt.envs.reset()
        start_p = opt.envs.best_start[0]
        start_c = opt.envs.best_start[1]
        epsilon = max(0.001, epsilon * epsilon_decay)

        num_episodes += 1
        if args.limit_runs and args.max_num_episodes < num_episodes:
            print(num_episodes, ' episodes done. Switching Task...')
            num_episodes = 0
            log_counts[fast_slow_switch] += 1
            fast_slow_switch = (fast_slow_switch + 1) % 2
            transfer_request = transfer_requests[fast_slow_switch]
            if fast_slow_switch == 1:
                os.system("mv /home/cc/rl-optimizer/time.log /home/cc/rl-optimizer/short-time-%d" % log_counts[0])
                os.system("sudo /home/cc/wondershaper/wondershaper -a eno1 -d 1000000")
                print('Switching to slow transfers; Episode', num_episodes)
            else:
                os.system("mv /home/cc/rl-optimizer/time.log /home/cc/rl-optimizer/long-time-%d" % log_counts[1])
                os.system("sudo /home/cc/wondershaper/wondershaper -c -a eno1")
                print('Switching to fast transfers; Episode', num_episodes)
            agent.true_delete(delete_op)
            scheduler.shutdown()
            scheduler = BackgroundScheduler()
        else:
            print('Starting Episode', num_episodes, '; Slow?', fast_slow_switch)

        schedule_thread = ScheduleTransfer().start()
        print("Reset to:", (start_p, start_c))
    return '', 204
