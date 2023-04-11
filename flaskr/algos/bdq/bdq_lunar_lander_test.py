import gymnasium
from pettingzoo.butterfly import cooperative_pong_v5
from agents import BDQAgent
from algos.global_memory import ReplayBuffer
import torch
from algos.global_utils import evaluate_policy
import numpy as np

BATCH_SIZE = 64
MAX_STEPS = 999
# ENV_NAME="LunarLander-v2"
ENV_NAME = "MountainCar-v0"

if __name__ == "__main__":
    episode_rewards = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0)
    env = gymnasium.make(ENV_NAME, render_mode=None)
    # print(env.action_space.shape[0])
    # print(env.observation_space.shape)
    # print(type(env.action_space))
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = BDQAgent(state_dim=state_dim, num_actions=1, action_dims=[
                     action_dim], device=device)
    replay_buffer = ReplayBuffer(
        state_dimension=state_dim, action_dimension=action_dim)
    episodes = 2000
    state = env.reset()[0]
    # env.render()
    for episode in range(episodes):
        episode_reward = 0
        state = env.reset()[0]
        terminated = False
        truncated = False
        step_ct = 0

        eps_actions = []
        while not (terminated or truncated):
            action = agent.select_action(state)
            action = action[0]
            eps_actions.append(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, terminated)
            state = next_state
            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)

            step_ct += 1
            if step_ct > MAX_STEPS:
                break
            episode_reward += reward
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                print("Episode: ", episode, " reward=", episode_reward)
            if np.mean(episode_rewards[-10:]) > 200:
                break

        agent.update_epsilon()
        print(eps_actions)

    print("Episode avg", np.mean(episode_rewards[-10:]))
    agent.save_checkpoint(ENV_NAME)
    env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0,
                         render_mode="human")
    print("Total Mean reward from evaluate: ", evaluate_policy(
        policy=agent, env=env, seed=42, eval_episodes=20))
