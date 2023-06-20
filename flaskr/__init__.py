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
from .agent import *

from .middleware import OptimizerMap
import threading
# import requests
# import json
import pickle

import random

scheduler = BackgroundScheduler()

optim_map = OptimizerMap()
# with open('transfer.json') as f:
#     transfer_requests = [json.load(f)]
#
# with open('slow-transfer.json') as f:
#     transfer_requests.append(json.load(f))
#
# with open('slower-transfer.json') as f:
#     transfer_requests.append(json.load(f))

transfer_request = args.start_cmd[0]

start_p = 2
start_c = 2
epsilon = 1.
epsilon_decay = 0.93325
sample_space = [i for i in range(2, 7)]

num_episodes = 0
fast_slow_switch = 0
log_counts = [0, 0, 0]


class ScheduleTransfer(threading.Thread):
    def __init__(self):
        super(ScheduleTransfer, self).__init__()

    def run(self):
        for f in args.file_path:
            try:
                remove(f)
            except:
                pass
        time.sleep(45.)

        # Inject Parameters
        os.system(transfer_request % (start_c, start_p))

        # transfer_request['options']['concurrencyThreadCount'] = start_c
        # transfer_request['options']['parallelThreadCount'] = start_p

        # r = requests.post(
        #     "http://192.5.87.31:8092/api/v1/transfer/start",
        #     json=transfer_request
        # )
        # print(r.status_code, r.reason)


schedule_thread = None


def at_exit():
    # scheduler.shutdown()
    if schedule_thread:
        schedule_thread.join()
    optim_map.clean_all()


# atexit.register(at_exit)

app = Flask(__name__)
start = time.time()

@app.route('/optimizer/create/training', methods=['POST'])
def create_optimizer_training():
    if request.method == 'POST':
        json_dict = request.json
        print(json_dict)
        create_opt = CreateOptimizerRequest(json_dict['nodeId'], json_dict['optimizerType'], json_dict['oldJobId'])
        print(create_opt.__str__())

@app.route('/optimizer/create', methods=['POST'])
def create_optimizer():
    global start
    if request.method == 'POST':
        json_dict = request.json
        print(json_dict)
        create_opt = CreateOptimizerRequest(json_dict['nodeId'], json_dict['maxConcurrency'],
                                            json_dict['maxParallelism'], json_dict['maxPipelining'],
                                            json_dict['maxChunkSize'], json_dict['optimizerType'], json_dict['fileCount'], json_dict['jobId'], json_dict['dbType'])
        print(create_opt.__str__())
        if 'launch_job' in json_dict:
            create_opt.set_launch_job(json_dict['launch_job'])
        if create_opt.optimizerType == optim_map.vda2c:
            override = None
            if fast_slow_switch > 0:
                override = args.bandwidth_restriction[fast_slow_switch]
            schedule = optim_map.create_optimizer(create_opt, override_max=override)
            opt = optim_map.get_optimizer(create_opt.node_id)
            start = time.time()
            print("Start Time:", start)
            if schedule:
                scheduler.add_job(opt.envs.fetch_and_train, trigger='interval', seconds=15)
                scheduler.start()

        elif create_opt.optimizerType == optim_map.bo:
            optim_map.create_optimizer(create_opt)

        elif create_opt.optimizerType == optim_map.maddpg:
            #creates the optimizer maddpg
            optim_map.create_optimizer(create_opt)

        elif create_opt.optimizerType == optim_map.ddpg:
            optim_map.create_optimizer(create_opt)

        return ('', 204)

@app.route('/optimizer/parameters', methods=['GET'])
def input_to_optimizer():
    if request.method == 'GET':
        jd = request.json
        input_operation = InputOptimizerRequest(jd['nodeId'], jd['throughput'], jd['rtt'], jd['concurrency'],
                                                jd['parallelism'], jd['pipelining'], jd['chunkSize'])
        print(input_operation.__str__())
        return optim_map.input_optimizer(input_operation), 200


@app.route('/optimizer/delete', methods=['POST'])
def delete_optimizer():
    end = time.time()
    if request.method == 'POST':
        jd = request.json
        delete_op = DeleteOptimizerRequest(jd['nodeId'])
        if optim_map.node_id_to_optimizer[delete_op.node_id] == optim_map.bo:
            optim_map.delete_optimizer(delete_op)
        # elif optim_map.node_id_to_optimizer[delete_op.node_id] == optim_map.ddpg:
        #     optim_map.delete_optimizer(delete_op, args)
        elif optim_map.node_id_to_optimizer[delete_op.node_id] == optim_map.vda2c:
            print(delete_op.__str__())
            global epsilon
            global start_p
            global start_c
            global num_episodes
            global schedule_thread
            global transfer_request
            global fast_slow_switch
            global log_counts
            global scheduler

            thrput = args.job_size_Gbit / (end - start)  # MOVED TO ARGS
            print("^^^ JOB TIME", end - start, "SECONDS ^^^")
            print("THROUGHPUT:", thrput)
            with open('time.log', 'a') as f:
                f.write(str(end - start) + "," + str(thrput) + "\n")
            # print('Waiting for last entries...')
            # time.sleep(31.)
            optim_map.delete_optimizer(delete_req=delete_op, args=args)
            if args.wipe_optimizer_map:
                optim_map.true_delete(delete_op)
                scheduler.shutdown()
                scheduler = BackgroundScheduler()
            elif not args.evaluate and optim_map.node_id_to_optimizer[delete_op.node_id] == optim_map.vda2c:
                # optim_map.delete_optimizer(delete_op, args)
                opt = optim_map.get_optimizer(delete_op.node_id)
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
                    epsilon = 1
                    log_counts[fast_slow_switch] += 1
                    fast_slow_switch = (fast_slow_switch + 1) % args.number_tests
                    transfer_request = args.start_cmd[fast_slow_switch]
                    save_path = os.path.join(args.save_dir, args.algo)

                    if fast_slow_switch == 1:
                        os.system("mv /home/cc/rl-optimizer/time.log /home/cc/rl-optimizer/short-time-%d.log" % log_counts[0])
                        os.system("ssh serv ./restrict_link.sh eno1 2000 10000")

                        model_path = os.path.join(save_path, "unrestricted-%d.pt" % log_counts[0])
                        mv_command = "mv " + os.path.join(save_path, args.env_name + ".pt") + " " + model_path
                        os.system(mv_command)

                        with open(os.path.join(save_path, "unrest-sprout-%d.pkl" % log_counts[0]), 'wb') as pkl:
                            pickle.dump(opt.envs.parameter_dist_map, pkl)

                        print('Switching to slow transfers; Episode', num_episodes)
                    elif fast_slow_switch == 0:
                        os.system("mv /home/cc/rl-optimizer/time.log /home/cc/rl-optimizer/long-time-%d.log" % log_counts[1])
                        os.system("ssh serv ./free_link.sh eno1")

                        model_path = os.path.join(save_path, "restricted-%d.pt" % log_counts[1])
                        mv_command = "mv " + os.path.join(save_path, args.env_name + ".pt") + " " + model_path
                        os.system(mv_command)

                        with open(os.path.join(save_path, "rest-sprout-%d.pkl" % log_counts[1]), 'wb') as pkl:
                            pickle.dump(opt.envs.parameter_dist_map, pkl)

                        print('Switching to fast transfers; Episode', num_episodes)
                    else:  # dead code for now
                        os.system("mv /home/cc/rl-optimizer/time.log /home/cc/rl-optimizer/longer-time-%d.log" % log_counts[2])
                        os.system("sudo /home/cc/wondershaper/wondershaper -c -a eno1")
                        print('Switching to fast transfers; Episode', num_episodes)
                    optim_map.true_delete(delete_op)
                    scheduler.shutdown()
                    scheduler = BackgroundScheduler()
                else:
                    print('Starting Episode', num_episodes, '; Slow switch:', fast_slow_switch)

                schedule_thread = ScheduleTransfer().start()
                print("Reset to:", (start_p, start_c))
    return '', 204