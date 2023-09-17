import gymnasium
from agents import DDPGAgent
from memory import ReplayBuffer
import torch
import utils
import numpy as np

from tqdm import tqdm

BATCH_SIZE = 256
MAX_STEPS = 5000
ENV_NAME="LunarLanderContinuous-v2"
# ENV_NAME="MountainCarContinuous-v0"

enable_logging = True

if __name__ == "__main__":
    episode_rewards = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0)
    env = gymnasium.make(ENV_NAME, render_mode=None)
    # print(env.action_space.shape[0])
    # print(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device, max_action=1, discount=0.99)
    replay_buffer = ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
    episodes = 10000
    state = env.reset()[0]
    # env.render()

    if enable_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('./logs/' + ENV_NAME + '/')

    try:
        os.mkdir('./models')
    except Exception as e:
        pass
    
    # dist_width = 1.
    dist_width = 0.1
    for episode in tqdm(range(episodes), unit='eps'):
        episode_reward = 0
        state = env.reset()[0]
        terminated = False
        truncated = False
        step_ct = 0
        while not (terminated or truncated):
            if episode < 500:
                action = env.action_space.sample()
            else:
                action = (
                    agent.select_action(np.array(state)) + np.random.normal(0, dist_width, size=action_dim)
                ).clip(-1, 1)
                # dist_width *= 0.99948
            
            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, terminated)
            state = next_state
            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)

            step_ct += 1
            # if step_ct > MAX_STEPS:
            #     break
            episode_reward += reward
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                if enable_logging:
                    writer.add_scalar('Episode Reward', episode_reward, episode)

                # print("Episode: ", episode, " reward=", episode_reward)

        
        if np.mean(episode_rewards[-10:]) > 200:
            break

    print("Episode avg", np.mean(episode_rewards[-10:]))
    # agent.save_checkpoint('./models/'+ENV_NAME)
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0,
    #                      render_mode="human")
    # print("Total Mean reward from evaluate: ", utils.evaluate_policy(policy=agent, env=env, seed=42, eval_episodes=20))
