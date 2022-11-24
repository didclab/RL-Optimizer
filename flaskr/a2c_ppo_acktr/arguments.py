# import argparse
# from tkinter import N

import torch
import glob


class Args:
    def __init__(self):
        self.algo = 'vdac'
        self.gail = False
        self.gail_experts_dir = './gail_experts'
        self.gail_batch_size = 128
        self.gail_epoch = 5
        self.lr = 7e-4
        self.eps = 1e-5
        self.alpha = 0.99
        self.gamma = 0.99
        self.use_gae = False
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.seed = 1
        self.cuda_deterministic = False
        self.num_processes = 1
        self.num_steps = 5
        self.ppo_epoch = 4
        self.num_mini_batch = 32
        self.clip_param = 0.2
        self.log_interval = 10
        self.save_interval = 5
        self.eval_interval = None
        self.num_env_steps = 10e6
        self.env_name = 'Influx-v1'
        self.log_dir = '/tmp/gym/'
        self.save_dir = '/home/cc/rl-optimizer/flaskr/trained_models'
        self.no_cuda = False
        self.use_proper_time_limits = False
        self.recurrent_policy = False
        self.use_linear_lr_decay = False

        self.new_policy = True
        self.enable_vdac = True
        self.ping_interval = 30.
        self.limit_runs = True
        self.max_num_episodes = 119

        # self.file_path = '/home/didclab/Pictures/ubuntu-20.04.3-desktop-amd64(1).iso'
        # self.file_path = '/home/didclab/Pictures/moby.img'
        # self.file_path = glob.glob("/home/cc/dest/monty-*") # ['/media/didclab/edr/monty.img']
        self.file_path = ['/mnt/ramdisk/dest/monty-' + str(i) + '.dmg' for i in range(1, 33)]
        self.env_reg = 0.0078125
        self.starting_action = {
            'chunkSize': 68000000.0,  # set this to starting parameters
            'concurrency': 1.0,  # set this to starting parameters
            'parallelism': 1.0,  # set this to starting parameters
            'pipelining': 8.0,  # set this to starting parameters
        }
        self.bandwidth_restriction = [0.3125, 0.0625, 0.09375]

        self.number_tests = 2
        self.job_size_Gbit = 318
        self.start_cmd = [
            "python ~/odscli/onedatashare.py transfer http tacc-http \"/\" -f \"\" vfs " +
            "elvisdav@buffalo.edu-didclab-elvis-uc \"/mnt/ramdisk/dest/\" --concurrency=%d --parallel=%d --pipesize=8" +
            " --chunksize=67108864 --compress=false --retry=1",

            "python ~/odscli/onedatashare.py transfer http tacc-http \"/\" -f \"\" vfs " +
            "elvisdav@buffalo.edu-didclab-elvis-uc \"/mnt/ramdisk/dest/\" --concurrency=%d --parallel=%d --pipesize=8" +
            " --chunksize=67108864 --compress=false --retry=1",

            "python ~/odscli/onedatashare.py transfer http tacc-http \"/\" -f \"\" vfs " +
            "elvisdav@buffalo.edu-didclab-elvis-uc \"/mnt/ramdisk/dest/\" --concurrency=%d --parallel=%d --pipesize=8" +
            " --chunksize=67108864 --compress=false --retry=1"
        ]

        self.cuda = False


def get_args():
    args = Args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'vdac']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
