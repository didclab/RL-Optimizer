import gym
import numpy as np


def evaluate_policy(policy, env, seed, eval_episodes=10, render=False):
    avg_reward = 0.
    for _ in range(eval_episodes):
        options = {'launch_job':True}
        state, done = env.reset(options=options), False
        while not done:
            action = policy.select_action(np.array(state))
            if render:
                env.render()
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward
