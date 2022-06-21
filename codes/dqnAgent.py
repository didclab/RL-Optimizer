import torch
from torch import nn
import gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=100000
TARGET_UPDATE_FREQ=1000

def print_list_action(lst,env):
  action_dictionary=dict()
  for action in range(env.action_space.n):
    value=env.actions[action]
    action_dictionary[action]=value
  return_list=[]
  for i in lst:
      return_list.append(action_dictionary[i])
  return return_list


class Network(nn.Module):
  def __init__(self,env):
      super().__init__()
      in_features=int(np.prod(env.obs_shape))
      self.net=nn.Sequential(
      nn.Linear(in_features,1024),
      nn.Tanh(),
      # nn.Linear(1024,512),
      # nn.Tanh(),
      nn.Linear(1024, env.action_space.n))

  def forward(self,x):
    return self.net(x)

  def act(self,obs):
    obs_t=torch.as_tensor(obs,dtype=torch.float32)
    q_values=self(obs_t.unsqueeze(0))
    max_q_index=torch.argmax(q_values,dim=1)[0]
    action=max_q_index.detach().item()
    return action


class DQNAgent():
  def __init__(self,env,GAMMA=0.99,
                  BATCH_SIZE=1024,
                  BUFFER_SIZE=6000,
                  MIN_REPLAY_SIZE=1000,
                  EPSILON_START=1.0,
                  EPSILON_END=0.2,
                  EPSILON_DECAY=100000,
                  TARGET_UPDATE_FREQ=1000,
                  LOSS_TYPE='L1'):
    self.env=env
    self.GAMMA=GAMMA
    self.BATCH_SIZE=BATCH_SIZE
    self.BUFFER_SIZE=BUFFER_SIZE
    self.MIN_REPLAY_SIZE=MIN_REPLAY_SIZE
    self.EPSILON_START=EPSILON_START
    self.EPSILON_DECAY=EPSILON_DECAY
    self.EPSILON_END=EPSILON_END
    self.TARGET_UPDATE_FREQ=TARGET_UPDATE_FREQ
    self.replay_buffer=deque(maxlen=self.BUFFER_SIZE)
    self.WRITER=SummaryWriter('runs/DQN_'+str(time.strftime("%D_%H_%M_%S",time.localtime())))
    self.rew_buffer=deque([0.0],maxlen=100)
    self.episode_reward=0.0
    self.LOSS_TYPE=LOSS_TYPE
    self.online_net=Network(env)
    self.target_net=Network(env)
    self.target_net.load_state_dict(self.online_net.state_dict())
    self.key_max_action={}
    for key in range(len(self.env.group_keys)):
        for action in range(self.env.action_space.n):
            obs=self.env.reset_group_specific(key)
            new_obs,rew,done, _ =self.env.step(action)
            if rew == 1.0:
                self.key_max_action[key]=action
                break
    self.optimizer=torch.optim.Adam(self.online_net.parameters(),lr=1e-3)
    self.reward_per_episode=[]
    self.epsilon_per_episode=[]

  def warming_replay_buffer_prioritize(self):
    print("warming up memory with prioritization")
    for key in range(len(self.env.group_keys)):
      obs=self.env.reset_group_specific(key)
      for action in range(self.env.action_space.n):
        obs,_,_,_=self.env.step(action)
        # obs=self.env.reset_group_specific(key)
        new_obs,rew,done, _ =self.env.step(self.key_max_action[key])
        transition=(obs,self.key_max_action[key],rew,done,new_obs)
        self.replay_buffer.append(transition)
        obs=self.env.reset_group_specific(key)

  def net_tensorboard(self):
    transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
    obses=np.asarray([t[0] for t in transitions])
    actions=np.asarray([t[1] for t in transitions])
    rews=np.asarray([t[2] for t in transitions])
    dones=np.asarray([t[3] for t in transitions])
    new_obses=np.asarray([t[4] for t in transitions])
    obses_t=torch.as_tensor(obses,dtype=torch.float32)
    actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
    rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
    dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
    new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)
    self.WRITER.add_graph(self.online_net,obses_t)


  def warming_replay_buffer(self):
    obs=self.env.reset()
    print("warming up memory")
    for _ in range(self.MIN_REPLAY_SIZE):
      action=self.env.action_space.sample()
      new_obs,rew,done, _ =self.env.step(action)
      transition=(obs,action,rew,done,new_obs)
      self.replay_buffer.append(transition)
      obs=new_obs
      if done:
        print("episode ends")
        obs=self.env.reset()

  def training_endlessly(self):
    obs=self.env.reset()
    for step in itertools.count():
    #for step in range(TRAINING_STEPS):
        epsilon=np.interp(step,[0,self.EPSILON_DECAY],[self.EPSILON_START,self.EPSILON_END])
        rnd_sample=random.random()

        ####following implements epsilon-greedy policy
        if rnd_sample <=epsilon:
            action=self.env.action_space.sample()
        else:
            action=self.online_net.act(obs)

        new_obs,rew,done, _ =self.env.step(action)
        transition= (obs,action,rew,done,new_obs)
        if rew>0:
          self.replay_buffer.append(transition)
        obs=new_obs
        self.episode_reward+=rew
        if done:
            obs=self.env.reset()
            self.rew_buffer.append(self.episode_reward)
            self.reward_per_episode.append(self.episode_reward)
            self.epsilon_per_episode.append(epsilon)
            self.episode_reward=0.0
            print("episode ends")


        #### Satrt gradient Step
        transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
    #     print(transitions)
        obses=np.asarray([t[0] for t in transitions])
        actions=np.asarray([t[1] for t in transitions])
        rews=np.asarray([t[2] for t in transitions])
        dones=np.asarray([t[3] for t in transitions])
        new_obses=np.asarray([t[4] for t in transitions])

        obses_t=torch.as_tensor(obses,dtype=torch.float32)
        actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
        rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
        dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
        new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)

        #compute Targets
        target_q_values=self.target_net(new_obses_t)
        max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
        targets=rews_t+GAMMA *(1-dones_t) * max_target_q_values

        # Compute Loss
        q_values=self.online_net(obses_t)
        action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
        loss=nn.functional.smooth_l1_loss(action_q_values,targets)

        ## Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target network
        if step % self.TARGET_UPDATE_FREQ ==0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        ##logging
        if step %1000 ==0:
            print()
            print('step', step)
            print('Avg Rew',np.mean(self.rew_buffer))

  def training(self,TRAINING_STEPS=2000):
    obs=self.env.reset()
    # for step in itertools.count():
    for step in range(TRAINING_STEPS):
      epsilon=np.interp(step,[0,self.EPSILON_DECAY],[self.EPSILON_START,self.EPSILON_END])
      action=self.online_net.act(obs)
      new_obs,rew,done, _ =self.env.step(action)
      transition= (obs,action,rew,done,new_obs)
      obs=new_obs
      self.episode_reward+=rew

      if done:
        obs=self.env.reset()
        self.rew_buffer.append(self.episode_reward)
        self.WRITER.add_scalar("reward/train", self.episode_reward,step)
        self.reward_per_episode.append(self.episode_reward)
        self.epsilon_per_episode.append(epsilon)
        self.episode_reward=0.0
        print("episode ends")
      #### Satrt gradient Step
      transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
  #     print(transitions)
      obses=np.asarray([t[0] for t in transitions])
      actions=np.asarray([t[1] for t in transitions])
      rews=np.asarray([t[2] for t in transitions])
      dones=np.asarray([t[3] for t in transitions])
      new_obses=np.asarray([t[4] for t in transitions])

      obses_t=torch.as_tensor(obses,dtype=torch.float32)
      actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
      rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
      dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
      new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)

      #compute Targets
      target_q_values=self.target_net(new_obses_t)
      max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
      targets=rews_t+GAMMA *(1-dones_t) * max_target_q_values

      # Compute Loss
      q_values=self.online_net(obses_t)
      action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
      loss=nn.functional.smooth_l1_loss(action_q_values,targets)
      self.WRITER.add_scalar("loss/train", loss, step)
      ## Gradient Descent
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      # update target network
      if step % self.TARGET_UPDATE_FREQ ==0:
        self.target_net.load_state_dict(self.online_net.state_dict())
      ##logging
      if step %1000 ==0:
        print()
        print('step', step)
        print('Avg Rew',np.mean(self.rew_buffer))

  def training_only_from_replay_buffer(self,TRAINING_STEPS=2000):
    for step in range(TRAINING_STEPS):
      for i in range (0,len(self.online_net.net),2):
        self.WRITER.add_histogram(f"layer{i}_{self.LOSS_TYPE}", self.online_net.net[i].weight.flatten(), global_step=step, bins='tensorflow')
      #### Satrt gradient Step
      # weight_histograms(self.WRITER, step, self.online_net)
      # print("weight_histogram has been called")
      transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
      obses=np.asarray([t[0] for t in transitions])
      actions=np.asarray([t[1] for t in transitions])
      rews=np.asarray([t[2] for t in transitions])
      dones=np.asarray([t[3] for t in transitions])
      new_obses=np.asarray([t[4] for t in transitions])

      obses_t=torch.as_tensor(obses,dtype=torch.float32)
      actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
      rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
      dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
      new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)

      #compute Targets
      target_q_values=self.target_net(new_obses_t)
      max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
      targets=rews_t+GAMMA *(1-dones_t) * max_target_q_values

      # Compute Loss
      q_values=self.online_net(obses_t)
      action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
      loss=nn.functional.smooth_l1_loss(action_q_values,targets)

      # loss=nn.functional.l1_loss(action_q_values,targets)

      self.WRITER.add_scalar("loss/train", loss, step)
      ## Gradient Descent
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      # update target network
      if step % self.TARGET_UPDATE_FREQ ==0:
        self.target_net.load_state_dict(self.online_net.state_dict())
      ##logging
      if step %1000 ==0:
        print()
        print('step', step)
        print('loss',loss)


  def save_model(self):
    name='agent_UF_'+str(TARGET_UPDATE_FREQ)+'_BS_'+str(self.BATCH_SIZE)+'_B_'+str(self.GAMMA)
    torch.save(self.online_net, name+"_online_net")
    torch.save(self.target_net, name+"_target_net")

  def load_model(self,online_net,target_net):
    self.online_net = torch.load("online_net")
    self.target_net = torch.load("target_net")

  def plot_training_curve(self):
    reward_epsilon_values=[]
    reward_epsilon_values.append(self.reward_per_episode)
    reward_epsilon_values.append(self.epsilon_per_episode)
    labels = ["reward", "epsilon"]
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('reward-episodes & epsilon-episodes graph DQN for selectedGroup')
    for i,ax in enumerate(axs):
        axs[i].plot(reward_epsilon_values[i],label=labels[i])
        axs[i].legend(loc="lower right")
    plt.show()

  def writer_flush(self):
    self.WRITER.add_graph(self.online_net)
    self.WRITER.flush()
    self.WRITER.close()



