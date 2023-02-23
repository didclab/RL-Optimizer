import torch
import numpy as np
import os
from threading import Thread
from .ods_env import ods_influx_gym_env
from flaskr import classes
from .algos.ddpg import agents
from .algos.ddpg import memory
from .algos.ddpg import utils
import flaskr.ods_env.ods_helper as ods


class Trainer(object):
    def __init__(self, create_opt_request=classes.CreateOptimizerRequest, max_episodes=100, batch_size=64, update_policy_time_steps=20):
        self.obs_cols = []
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.create_opt_request = create_opt_request
        self.env = ods_influx_gym_env.InfluxEnv(create_opt_req=create_opt_request,time_window="-1d")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = agents.DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=self.device, max_action=None)
        self.replay_buffer = memory.ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
        self.save_file_name = f"DDPG_{'influx_gym_env'}"
        try:
            os.mkdir('./models')
        except Exception as e:
            pass
        self.worker_thread = Thread(target=self.train, args=(max_episodes, batch_size, update_policy_time_steps))


    #creates a worker to start running training and then joins when the model is done training
    #runs evaluate on main thread as that should be super fast.
    def thread_train(self):
        #check if job is running
        running, meta = ods.query_if_job_running(self.create_opt_request.job_id)
        if not running:
            #submit job if it is not running
            ods.submit_transfer_request(meta)
        #start worker threads to begin training
        self.worker_thread.start()
        self.worker_thread.join()
        return self.evaluate()


    #the very first call to train already has a job running
    def train(self, max_episodes=100, batch_size=64, update_policy_time_steps=20, launch_job=False):
        options = {'launch_job': launch_job}
        state = self.env.reset(options=options) #no seed the first time
        episode_reward = 0
        episode_num = 0
        episode_ts = 0
        episode_rewards = []
        #iterate until we hit max jobs
        while episode_num < max_episodes:
            action = (self.agent.select_action(np.array(state)))
            prev_state, prev_reward, terminated, _ = self.env.step(action)
            episode_ts+=1
            self.replay_buffer.add(state, action, prev_state, prev_reward, terminated)
            state = prev_state
            episode_reward += prev_reward
            if terminated:
                self.env.reset(options={'launch_job': True})
                print("Episode reward: {}", episode_reward)
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_ts = 0

            if episode_ts % update_policy_time_steps==0:
                self.agent.train(self.replay_buffer, batch_size)

        return episode_rewards

    def evaluate(self):
        avg_reward = utils.evaluate_policy(policy=self.agent, env=self.env)
        self.agent.save_checkpoint(f"./models/{self.save_file_name}")
        return avg_reward






