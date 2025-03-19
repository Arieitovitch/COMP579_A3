import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import json
import gymnasium as gym
import ale_py 

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.uniform_(layer.weight, -0.001, 0.001)
            nn.init.uniform_(layer.bias, -0.001, 0.001)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ExpectedSARSA(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ExpectedSARSA, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.uniform_(layer.weight, -0.001, 0.001)
            nn.init.uniform_(layer.bias, -0.001, 0.001)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, n_observations, n_actions, lr=0.001, epsilon=0.1, algorithm="DQN", use_replay_buffer=False):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.use_replay_buffer = use_replay_buffer
        
        if algorithm == "DQN":
            self.model = DQN(n_observations, n_actions)
        elif algorithm == "ExpectedSARSA":
            self.model = ExpectedSARSA(n_observations, n_actions)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = [] if use_replay_buffer else None
        self.buffer_capacity = 1000000
        self.batch_size = 64

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs / 255.0, dtype=torch.float32).unsqueeze(0) 
                return torch.argmax(self.model(obs_tensor)).item()

    def update(self):
        if self.use_replay_buffer:
            if len(self.replay_buffer) < self.batch_size:
                return
            batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = self.replay_buffer
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        if self.algorithm == "DQN":
            next_q_values = self.model(next_states).max(1)[0]
        elif self.algorithm == "ExpectedSARSA":
            next_q_values = (self.model(next_states).exp() / self.model(next_states).exp().sum(dim=1, keepdim=True)).sum(dim=1)
        
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
        
        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        if self.use_replay_buffer:
            if len(self.replay_buffer) >= self.buffer_capacity:
                self.replay_buffer.pop(0)
            self.replay_buffer.append((state, action, reward, next_state, done))
        else:
            self.replay_buffer = [(state, action, reward, next_state, done)]


def train(env_name, episodes=1000, render=True, algorithm="DQN", use_replay_buffer=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, algorithm=algorithm, use_replay_buffer=use_replay_buffer)
    results = []
    
    for episode in range(episodes):
        state = env.reset()[0] / 255.0
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state / 255.0 
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        
        results.append({"episode": episode, "reward": total_reward, "algorithm": algorithm, "replay_buffer": use_replay_buffer})
        print(f"Episode {episode + 1}: Total Reward: {total_reward} | Algorithm: {algorithm} | Replay Buffer: {use_replay_buffer}")
    
    env.close()
    
    with open(f"training_results_{algorithm}_buffer_{use_replay_buffer}.json", "w") as f:
        json.dump(results, f)
    print(f"Results saved to training_results_{algorithm}_buffer_{use_replay_buffer}.json")
    
for env_name in ["ALE/Assault-ram-v5"]:
    for algorithm in ["DQN", "ExpectedSARSA"]:
        for use_replay_buffer in [True, False]:
            train(env_name, episodes=1000, render=True, algorithm=algorithm, use_replay_buffer=use_replay_buffer)

for env_name in ["Acrobot-v1"]:
    for algorithm in ["DQN", "ExpectedSARSA"]:
        for use_replay_buffer in [True, False]:
            train(env_name, episodes=1000, render=True, algorithm=algorithm, use_replay_buffer=use_replay_buffer)



