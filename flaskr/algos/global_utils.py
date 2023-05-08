import gym
import numpy as np


def evaluate_policy(policy, env, seed, eval_episodes=10, render=False):
    state = env.reset()[0]
    env.render()
    episodes_reward = []
    terminated = False
    for _ in range(eval_episodes):

        state = env.reset()[0]
        episode_reward = 0
        while not terminated:
            action = policy.select_action(np.array(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
        episodes_reward.append(episode_reward)

    return np.mean(episodes_reward)
