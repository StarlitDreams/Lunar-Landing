import os
import random
import numpy as np
import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions import Categorical
from collections import deque, namedtuple
import gymnasium as gym

class Network(nn.Module):
    def __init__ (self,state_size,action_size, seed=42) -> None:
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)
    
    def forward(self,state) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(state))
        return self.fc3(x)

    
env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape[0]
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

learning_rate = 5e-4
mini_batch_size = 100
gamma = 0.99
replay_buffer_size = int(1e6)
tao = 1e-3

class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.device = torch.device("cuda:0")
        self.capacity = capacity
        self.memory = []
        
    def push(self,event) -> None:
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self,batch_size) -> torch.Tensor:
        experiences = random.sample(self.memory, k = batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None].astype(np.uint8))).float().to(self.device)
        return next_states, actions, rewards, dones
        
    
class Agent():
    def __init__(self,state_size,action_size) -> None:
        self.device = torch.device("cuda:0")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork =  Network(state_size,action_size).to(self.device)
        self.target_qnetwork = Network(state_size,action_size).to(self.device)
        self.optimizer= optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step=0

    def step(self,state,action,reward,next_state,done) -> None:
        self.memory.push((state,action,reward,next_state,done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > mini_batch_size:
                experiences = self.memory.sample(mini_batch_size)
                self.learn(experiences,gamma) 
        
    def act(self,state,epsilon=0.0) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train() 
        if(random.random() > epsilon):
            return np.argmax(action_values.cpu().data.numpy()) 
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,experiences,gamma) -> None:
        states, next_states, actions, rewards, dones = experiences
        next_q_targets= self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma *next_q_targets * 1- dones )
        q_expected = self.local_qnetwork(states).gather(1,actions)
        loss = F.mse_loss(q_expected,q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork,self.target_qnetwork,tao)
        
    def soft_update(self,local_model,target_model,tao) -> None:
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tao*local_param.data + (1.0-tao)*target_param.data)
    
    
agent = Agent(state_size,number_actions)

number_episodes = 200000
max_t = 10000
epsilon_start = 1.0
eplison_decay = 0.995
epsilon_end = 0.01
epsilon = epsilon_start
scores_deque = deque(maxlen=100)

for episode in range(1,number_episodes+1):
    state,_ = env.reset()
    score=0
    for t in range(max_t):
        action = agent.act(state,epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.step(state,action,reward,next_state,done)
        state = next_state
        score += reward
        if done:
            break