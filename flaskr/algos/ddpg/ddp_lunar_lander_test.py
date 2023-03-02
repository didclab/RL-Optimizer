import gymnasium
from agents import DDPGAgent
from memory import ReplayBuffer
import torch

BATCH_SIZE = 64
MAX_STEPS = 400

if __name__ == "__main__":
    episode_rewards = []
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    env = gymnasium.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = DDPGAgent(state_dim=env, action_dim=action_dim, device=device, max_action=None)
    replay_buffer = ReplayBuffer(state_dimension=state_dim, action_dimension=action_dim)
    episodes = 5
    state = env.reset()
    env.render(mode="human")
    for episode in range(episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        step_ct = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)

            step_ct += 1
            if step_ct > MAX_STEPS:
                break

            episode += reward

            if done:
                episode_rewards.append(episode_reward)
