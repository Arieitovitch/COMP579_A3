import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import json
import gymnasium as gym
import ale_py 
import matplotlib.pyplot as plt
import glob

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
            next_q_values = self.model(next_states)
            
            # Compute policy probabilities for epsilon-greedy
            policy_probs = torch.ones_like(next_q_values) * (self.epsilon / self.n_actions)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)
            policy_probs.scatter_(1, best_actions, 1 - self.epsilon + (self.epsilon / self.n_actions))

            # Compute expected Q-value
            next_q_values = (next_q_values * policy_probs).sum(dim=1)

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


def train(env_name, episodes=1000, render=False, algorithm="DQN", use_replay_buffer=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, algorithm=algorithm, use_replay_buffer=use_replay_buffer)
    results = []
    
    for episode in range(episodes):
        state = env.reset()[0] / 255.0 if "ram" in env_name else env.reset()[0]  # Normalize if needed
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            if "ram" in env_name:
                next_state = next_state / 255.0  # Normalize for Atari RAM-based environments

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward  # Accumulate rewards without modification

        results.append({"episode": episode, "reward": total_reward, "algorithm": algorithm, "replay_buffer": use_replay_buffer})
        print(f"Episode {episode + 1}: Total Reward: {total_reward} | Algorithm: {algorithm} | Replay Buffer: {use_replay_buffer} | Environment: {env_name}")
    
    env.close()
    
    safe_env_name = env_name.replace("/", "_")  # Fix file name
    with open(f"training_results_{safe_env_name}_{algorithm}_buffer_{use_replay_buffer}.json", "w") as f:
        json.dump(results, f)
    print(f"✅ Results saved to training_results_{safe_env_name}_{algorithm}_buffer_{use_replay_buffer}.json")





    
for env_name in ["ALE/Assault-ram-v5"]:
    for algorithm in ["DQN", "ExpectedSARSA"]:
        for use_replay_buffer in [True, False]:
            train(env_name, episodes=10, render=False, algorithm=algorithm, use_replay_buffer=use_replay_buffer)

for env_name in ["Acrobot-v1"]:
    for algorithm in ["DQN", "ExpectedSARSA"]:
        for use_replay_buffer in [True, False]:
            train(env_name, episodes=10, render=False, algorithm=algorithm, use_replay_buffer=use_replay_buffer)


colors = {"DQN": "green", "ExpectedSARSA": "red"}
linestyles = {0.01: "solid", 0.001: "dashed", 0.0001: "dotted"}


def load_results():
    """
    Loads training results from JSON files and ensures proper formatting.
    Returns:
        data (dict): Dictionary with keys as experiment configurations and values as lists of runs.
    """
    files = glob.glob("training_results_*.json")
    data = {}

    for file in files:
        with open(file, "r") as f:
            try:
                results = json.load(f)
                
                # Debugging print to check the structure of each file
                print(f"🔹 Loaded {file}: Type={type(results)}, Length={len(results) if isinstance(results, list) else 'N/A'}")

                # Ensure results are a list
                if not isinstance(results, list):
                    print(f"⚠️ Warning: {file} does not contain a list. Skipping.")
                    continue

                key = file.replace("training_results_", "").replace(".json", "")

                if key not in data:
                    data[key] = []

                data[key].extend(results)

            except json.JSONDecodeError:
                print(f"⚠️ Warning: Failed to load {file}, skipping.")

    return data



def process_data(results):
    if not results or not isinstance(results, list):
        raise ValueError("No valid results available to process.")

    # Ensure each run is a list of dictionaries
    cleaned_results = []
    for run in results:
        if isinstance(run, dict):
            print(f"⚠️ Unexpected format (dict instead of list), skipping: {run}")
            continue  # Skip incorrectly formatted data

        if isinstance(run, list) and all(isinstance(entry, dict) for entry in run):
            cleaned_results.append(run)
        else:
            print(f"⚠️ Skipping invalid run format: {type(run)}")

    if not cleaned_results:
        raise ValueError("All runs are incorrectly formatted or empty.")

    # Get the longest episode count
    num_episodes = max(len(run) for run in cleaned_results)

    # Initialize an array to store rewards from all runs, using NaN for missing values
    all_rewards = np.full((len(cleaned_results), num_episodes), np.nan)

    # Populate rewards from each run
    for i, run in enumerate(cleaned_results):
        rewards = [entry.get("reward", np.nan) for entry in run]  # Use `.get()` to avoid errors
        all_rewards[i, : len(rewards)] = rewards  

    # Compute mean and standard deviation while ignoring NaN values
    mean_rewards = np.nanmean(all_rewards, axis=0)
    std_rewards = np.nanstd(all_rewards, axis=0)

    return mean_rewards, std_rewards





def plot_results(data, env_name):
    plt.figure(figsize=(10, 6))
    
    plotted_anything = False

    for key, results in data.items():
        if env_name.replace("/", "_") not in key:  # Fix name issues
            continue

        algorithm = "DQN" if "DQN" in key else "ExpectedSARSA"
        use_replay_buffer = "True" in key
        linestyle = "solid"

        mean_rewards, std_rewards = process_data(results)

        if mean_rewards.size == 0:
            print(f"⚠️ Skipping {key} due to missing data.")
            continue

        label = f"{algorithm} {'(Replay Buffer)' if use_replay_buffer else '(No Replay Buffer)'}"
        plt.plot(mean_rewards, color=colors[algorithm], linestyle=linestyle, label=label)
        plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[algorithm], alpha=0.2)

        plotted_anything = True

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Training Performance for {env_name}")
    if plotted_anything:
        plt.legend()
    plt.grid(True)

    env_name_safe = env_name.replace("/", "_")  # Fix file naming
    plt.savefig(f"{env_name_safe}_performance.png")
    plt.show()



def plot_results(data, env_name):
    plt.figure(figsize=(10, 6))
    
    for key, results in data.items():
        if env_name not in key:
            continue
        
        algorithm = "DQN" if "DQN" in key else "ExpectedSARSA"
        use_replay_buffer = "True" in key
        linestyle = linestyles[0.01]

        mean_rewards, std_rewards = process_data(results)
        label = f"{algorithm} {'(Replay Buffer)' if use_replay_buffer else '(No Replay Buffer)'}"

        plt.plot(mean_rewards, color=colors[algorithm], linestyle=linestyle, label=label)
        plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[algorithm], alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Training Performance for {env_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{env_name}_performance.png")
    plt.show()


data = load_results()
for env in ["Acrobot-v1", "ALE/Assault-ram-v5"]:
    plot_results(data, env)
