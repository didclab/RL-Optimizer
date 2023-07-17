import gymnasium
from agents import GAEXAgent
from algos.global_memory import ReplayBuffer
import torch
# from algos.global_utils import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 256
MAX_STEPS = 5000
ENV_NAME="LunarLander-v2"
# ENV_NAME = "MountainCar-v0"
# ENV_NAME = "CartPole-v1"


if __name__ == "__main__":
    episode_rewards = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0)
    env = gymnasium.make(ENV_NAME, render_mode=None)
    # print(env.action_space.shape[0])
    # print(env.observation_space.shape)
    # print(type(env.action_space))
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = GAEXAgent(state_dim=state_dim, action_dims=[action_dim], device=device, decay=0.999593)
    replay_buffer = ReplayBuffer(state_dimension=state_dim, action_dimension=1, max_size=int(1e4))
    episodes = 25000
    state = env.reset()[0]
    # env.render()
    
    for _ in range(5000):
        state = env.reset()[0]
        terminated = False
        truncated = False
        step_ct = 0

        while not (terminated or truncated):
            action = np.random.randint(0, 4)
            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, terminated or truncated)
            state = next_state


    print("Finished warming up")
    ep_modulo = 0
    for episode in range(episodes):
        episode_reward = 0
        state = env.reset()[0]
        terminated = False
        truncated = False
        step_ct = 0

        # eps_actions = []
        while not (terminated or truncated):
            action = agent.select_action(state)
            action = action[0]
            # eps_actions.append(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, terminated or truncated)
            state = next_state
            # if replay_buffer.size > BATCH_SIZE:
            #     agent.train(replay_buffer, BATCH_SIZE)

            step_ct += 1
            if step_ct > MAX_STEPS:
                break
            episode_reward += reward
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                if episode_reward > -200:
                    print("Episode: ", episode, " reward=", episode_reward)

        if np.mean(episode_rewards[-10:]) > 200:
            break

        if replay_buffer.size > BATCH_SIZE and ep_modulo == 3:
            agent.train(replay_buffer, BATCH_SIZE)

        if ep_modulo < 3:
            ep_modulo += 1
        else:
            ep_modulo = 0
            
        agent.update_epsilon()
        # print(eps_actions)

    print("Episode avg", np.mean(episode_rewards[-10:]))

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(agent.G_loss,label="G")
    plt.plot(agent.D_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("gaex_gan_loss.png")
    
    # agent.save_checkpoint(ENV_NAME)
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0,
    #                      render_mode="human")
    # print("Total Mean reward from evaluate: ", evaluate_policy(
    #     policy=agent, env=env, seed=42, eval_episodes=20))
