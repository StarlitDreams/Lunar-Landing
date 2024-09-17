import os
import random
import numpy as np
import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque, namedtuple
import gymnasium as gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


env = gym.make('LunarLander-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 1e-3 
batch_size = 256     
gamma = 0.99
replay_buffer_size = int(1e5) 
tau = 1e-3
update_every = 4


class ReplayMemory(object):
    def __init__(self, capacity, state_size, device):
        self.capacity = capacity
        self.device = device
        self.state_size = state_size
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.uint8)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        index = self.position % self.capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        states = torch.from_numpy(self.states[idx]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).float().to(self.device)
        actions = torch.from_numpy(self.actions[idx]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).float().to(self.device)
        dones = torch.from_numpy(self.dones[idx]).float().to(self.device)
        return (states, next_states, actions, rewards, dones)

    def __len__(self):
        return self.size


class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size, state_size, self.device)
        self.t_step = 0
        self.criterion = nn.SmoothL1Loss()  
        self.scaler = torch.cuda.amp.GradScaler()  

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done) 
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experiences = self.memory.sample(batch_size)
                self.learn(experiences, gamma)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, next_states, actions, rewards, dones = experiences
        with torch.cuda.amp.autocast():
            next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (gamma * next_q_targets * (1 - dones))
            q_expected = self.local_qnetwork(states).gather(1, actions)
            loss = self.criterion(q_expected, q_targets)


        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.soft_update(self.local_qnetwork, self.target_qnetwork, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


agent = Agent(state_size, action_size)

number_episodes = 2000
max_t = 1000
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_end = 0.01
epsilon = epsilon_start
scores_deque = deque(maxlen=100)
all_scores = []

for episode in range(1, number_episodes + 1):
    state, _ = env.reset(seed=seed)
    score = 0
    for t in range(max_t):
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_deque.append(score)
    all_scores.append(score)
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print('\rEpisode {} \tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end="")
    if episode % 100 == 0:
        print('\rEpisode {} \tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
    if np.mean(scores_deque) >= 200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_deque)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v3')
