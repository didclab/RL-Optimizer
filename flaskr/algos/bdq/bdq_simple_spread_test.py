# import gymnasium
from pettingzoo.mpe import simple_spread_v2
from agents import BDQAgent
from algos.global_memory import ReplayBuffer
import torch
from algos.global_utils import evaluate_policy
import numpy as np

BATCH_SIZE = 64
MAX_STEPS = 100
NUM_AGENTS=2
HARD_NAME='agent_0'
NUM_EPISODES = 10000

ENV_NAME='simple_spread_v2'


if __name__ == "__main__":
    episode_rewards = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0)
    
    env = simple_spread_v2.env(N=NUM_AGENTS, local_ratio=0., max_cycles=MAX_STEPS, continuous_actions=False, render_mode=None)

    # print(env.action_space.shape[0])
    # print(env.observation_space.shape)
    # print(type(env.action_space))

    action_dim = env.action_space('agent_0').n
    state_dim = env.state_space.shape[0]
    agent = BDQAgent(state_dim=state_dim, action_dims=[action_dim, action_dim], num_actions=NUM_AGENTS, decay=0.9988, device=device)
    replay_buffer = ReplayBuffer(state_dimension=state_dim, action_dimension=NUM_AGENTS)
    
    episodes = NUM_EPISODES
    env.reset()
    state = env.state()
    # env.render()
    
    """
    POPULATE REPLAY BUFFER, RANDOMLY
    """

    for _ in range(BATCH_SIZE):
        env.reset()
        state = env.state()
        next_state = state

        terminated = False
        truncated = False
        step_ct = 0

        actions = {}
        reward = 0.
        for ag in env.agent_iter():
            
            if step_ct % NUM_AGENTS == 0:
                if step_ct > 0:
                    # global step took place
                    # record in buffer
                    flat_actions = [actions[ag] for ag in env.agents]
                    replay_buffer.add(state, flat_actions, next_state, reward, terminated or truncated)

                # refresh actions
                if terminated:
                    break

                state = next_state
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            _, reward, termination, truncated, _ = env.last()
            if termination:
                terminated = True
            
            if termination or truncated:
                act = None
            else:
                act = actions[ag]
            
            env.step(act)
            next_state = env.state()
            step_ct += 1
        
    
    # env.reset()
    # agent_action = [2, 1]
    # actions = {agent: agent_action[i] for i, agent in enumerate(env.agents)}
    # print(actions)
    # a = 1/0
    """
    TRAIN BDQ AGENT
    """

    for episode in range(episodes):
        episode_reward = 0
        env.reset()
        state = env.state()
        next_state = state

        terminated = False
        truncated = False
        step_ct = 0

        # eps_actions = []

        actions = {}
        reward = 0.
        for ag in env.agent_iter():
            
            if step_ct % NUM_AGENTS == 0:
                if step_ct > 0:
                    # global step took place
                    # record in buffer
                    flat_actions = [actions[ag] for ag in env.agents]
                    replay_buffer.add(state, flat_actions, next_state, reward, terminated or truncated)
                    episode_reward += reward

                # refresh actions
                if terminated:
                    break

                state = next_state
                agent_action = agent.select_action(state)
                actions = {agent: agent_action[i] for i, agent in enumerate(env.agents)}

            _, reward, termination, truncated, _ = env.last()
            
            if termination:
                terminated = True

            if termination or truncated:
                act = None
            else:
                act = actions[ag]

            env.step(act)
            next_state = env.state()

            step_ct += 1

            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)
            
        episode_rewards.append(episode_reward)
        print("Episode: ", episode, " reward=", episode_reward, " steps: ", step_ct, " terminate: ", terminated, 
            " truncated: ", truncated)
        if np.mean(episode_rewards[-10:]) > 195:
            break

        agent.update_epsilon()
        # print(eps_actions)

    print("Episode avg", np.mean(episode_rewards[-10:]))
    agent.save_checkpoint(ENV_NAME)
    # env = gymnasium.make(ENV_NAME, continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0,
    #                      render_mode="human")
    # print("Total Mean reward from evaluate: ", evaluate_policy(
    #     policy=agent, env=env, seed=42, eval_episodes=20))
