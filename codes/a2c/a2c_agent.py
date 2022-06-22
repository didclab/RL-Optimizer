from networks import *
import gym
import tqdm
from tqdm import tqdm_notebook
import numpy as np
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter

#discount factor for future utilities
DISCOUNT_FACTOR = 0.99
#number of episodes to run
NUM_EPISODES = 1000
#max steps per episode
MAX_STEPS = 10000
#score agent needs for environment to be solved
SOLVED_SCORE = 200
#device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class A2cAgent():
  def __init__(self,env,DISCOUNT_FACTOR=0.99,
              NUM_EPISODES=1000,
              MAX_STEPS=10000,
              SOLVED_SCORE=200,
              DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
              ):

    self.env=env
    self.DISCOUNT_FACTOR=DISCOUNT_FACTOR
    self.NUM_EPISODES=NUM_EPISODES
    self.MAX_STEPS=MAX_STEPS
    self.SOLVED_SCORE=SOLVED_SCORE
    self.DEVICE=DEVICE
    self.WRITER=SummaryWriter('runs/A2C_'+str(time.strftime("%H_%M_%S",time.localtime())))

    self.actor_network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(self.DEVICE)
    self.value_network = StateValueNetwork(self.env.observation_space.shape[0]).to(self.DEVICE)

    #Init optimizer
    self.policy_optimizer=optim.Adam(self.actor_network.parameters(),lr=1e-3)
    self.stateval_optimizer=optim.Adam(self.value_network.parameters(),lr=1e-3)
    #track scores
    self.scores = []
    #recent 100 scores
    self.recent_scores = deque(maxlen=50)
    # self.WRITER.add_graph('actor',self.actor_network,torch.from_numpy(self.env.reset()).float().unsqueeze(0).to(self.DEVICE))
    # self.WRITER.add_graph('critic',self.value_network,torch.from_numpy(self.env.reset()).float().unsqueeze(0).to(self.DEVICE))

  def training_agent(self,NUM_EPISODES=1000,MAX_STEPS=100,SOLVED_SCORE=95):
    #run episodes
    for episode in tqdm.notebook.tqdm(range(NUM_EPISODES)):
      #init variables
      state = self.env.reset()
      done = False
      score = 0
      I = 1
      #run episode, update online
      self.WRITER.add_histogram("actor_layer_input", self.actor_network.input_layer.weight.flatten(), global_step=episode, bins='tensorflow')
      self.WRITER.add_histogram("actor_layer_output", self.actor_network.output_layer.weight.flatten(), global_step=episode, bins='tensorflow')
      self.WRITER.add_histogram("critic_layer_input", self.value_network.input_layer.weight.flatten(), global_step=episode, bins='tensorflow')
      self.WRITER.add_histogram("critic_layer_output", self.value_network.output_layer.weight.flatten(), global_step=episode, bins='tensorflow')

      for step in range(MAX_STEPS):
          #get action and log probability
          action, lp = select_action(self.actor_network, state,self.DEVICE)
          #step with action
          new_state, reward, done, _ = self.env.step(action)
          #update episode score
          score += reward
          #get state value of current state
          state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
          state_val = self.value_network (state_tensor)
          #get state value of next state
          new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(self.DEVICE)
          new_state_val = self.value_network(new_state_tensor)
          #if terminal state, next state val is 0
          if done:
              new_state_val = torch.tensor([0]).float().unsqueeze(0).to(self.DEVICE)
          #calculate value function loss with MSE
          val_loss = F.mse_loss(reward + self.DISCOUNT_FACTOR * new_state_val, state_val)
          val_loss *= I

          #calculate policy loss
          advantage = reward + self.DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
          policy_loss = -lp * advantage
          policy_loss *= I
          #Backpropagate policy
          self.policy_optimizer.zero_grad()
          policy_loss.backward(retain_graph=True)
          self.policy_optimizer.step()

          #Backpropagate value
          self.stateval_optimizer.zero_grad()
          val_loss.backward()
          self.stateval_optimizer.step()
          #move into new state, discount I
          state = new_state
          I *= self.DISCOUNT_FACTOR
          self.WRITER.add_scalar("I", I, step)
          if done:
            break

      #append episode score
      print(f"reward from episode {episode} is {score}")
      self.scores.append(score)
      self.recent_scores.append(score)
      self.WRITER.add_scalar("val_loss", val_loss, episode)
      self.WRITER.add_scalar("policy_loss", policy_loss, episode)
      self.WRITER.add_scalar("score/reward",score , episode)
      #early stopping if we meet solved score goal
      if np.array(self.recent_scores).mean() >= SOLVED_SCORE:
        break

  def save_model(self):
    name='agent_A2C_'+str(self.env)
    torch.save(self.actor_network, name+"actor_network")
    torch.save(self.value_network, name+"critic_network")

  def load_model(self,actor_network,value_network):
    self.actor_network = torch.load(actor_network)
    self.value_network = torch.load(value_network)
