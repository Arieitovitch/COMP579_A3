import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import time
import gymnasium as gym
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for results
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

#############################################
# PART 1: VALUE-BASED METHODS
#############################################

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        
        # Initialize parameters uniformly between -0.001 and 0.001
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.uniform_(layer.weight, -0.001, 0.001)
            nn.init.uniform_(layer.bias, -0.001, 0.001)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ValueBasedAgent:
    def __init__(self, n_observations, n_actions, lr=0.001, epsilon=0.1, 
                 algorithm="DQN", use_replay_buffer=False):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.use_replay_buffer = use_replay_buffer
        self.gamma = 0.99  # Discount factor
        
        # Initialize Q-network
        self.model = DQN(n_observations, n_actions).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Initialize replay buffer if needed
        if use_replay_buffer:
            self.replay_buffer = deque(maxlen=1000000)  # 1M capacity
        else:
            self.replay_buffer = []
            
        self.batch_size = 64
    
    def select_action(self, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()
    
    def update(self):
        if not self.use_replay_buffer and not self.replay_buffer:
            return 
        if self.use_replay_buffer:
            if len(self.replay_buffer) < self.batch_size:
                return 
            batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = [self.replay_buffer[-1]] * self.batch_size 

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.algorithm == "DQN":
                next_q = self.model(next_states).max(1)[0]
            elif self.algorithm == "ExpectedSARSA":
                next_q_values = self.model(next_states)
                probs = torch.ones(next_q_values.shape).to(device) * self.epsilon / self.n_actions
                best_actions = next_q_values.max(1)[1]
                for i in range(len(best_actions)):
                    probs[i, best_actions[i]] += (1 - self.epsilon)
                next_q = (next_q_values * probs).sum(1)

            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    
    def store_transition(self, state, action, reward, next_state, done):
        # Store transition in replay buffer
        if self.use_replay_buffer:
            self.replay_buffer.append((state, action, reward, next_state, done))
        else:
            self.replay_buffer = [(state, action, reward, next_state, done)]

#############################################
# PART 2: POLICY-BASED METHODS
#############################################

# class ActorNetwork(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(ActorNetwork, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 256)
#         self.layer2 = nn.Linear(256, 256)
#         self.layer3 = nn.Linear(256, n_actions)
        
#         # Initialize parameters uniformly between -0.001 and 0.001
#         for layer in [self.layer1, self.layer2, self.layer3]:
#             nn.init.uniform_(layer.weight, -0.001, 0.001)
#             nn.init.uniform_(layer.bias, -0.001, 0.001)
    
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)  # Logits for Boltzmann policy

# class ValueNetwork(nn.Module):
#     def __init__(self, n_observations):
#         super(ValueNetwork, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 256)
#         self.layer2 = nn.Linear(256, 256)
#         self.layer3 = nn.Linear(256, 1)
        
#         # Initialize parameters uniformly between -0.001 and 0.001
#         for layer in [self.layer1, self.layer2, self.layer3]:
#             nn.init.uniform_(layer.weight, -0.001, 0.001)
#             nn.init.uniform_(layer.bias, -0.001, 0.001)
    
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)  # State value

# class PolicyBasedAgent:
#     def __init__(self, n_observations, n_actions, lr=0.001, algorithm="REINFORCE", 
#                  fixed_temp=True, initial_temp=1.0, final_temp=0.1):
#         self.n_actions = n_actions
#         self.algorithm = algorithm
#         self.gamma = 0.99  # Discount factor
#         self.fixed_temp = fixed_temp
        
#         # Temperature parameters
#         self.initial_temp = initial_temp
#         self.final_temp = final_temp
#         self.current_temp = initial_temp
        
#         # Initialize policy network
#         self.actor = ActorNetwork(n_observations, n_actions).to(device)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
#         # Initialize value network (for Actor-Critic only)
#         if algorithm == "ActorCritic":
#             self.critic = ValueNetwork(n_observations).to(device)
#             self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
#         # Storage for episode data
#         self.reset_episode_data()
    
#     def reset_episode_data(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.next_states = []
#         self.dones = []
#         self.log_probs = []
        
#     def update_temperature(self, episode, total_episodes):
#         if not self.fixed_temp:
#             # Linear decay of temperature
#             self.current_temp = self.initial_temp - (self.initial_temp - self.final_temp) * (episode / total_episodes)
#             return self.current_temp
#         return self.initial_temp
    
#     def select_action(self, state):
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
#         # Get logits from policy network
#         logits = self.actor(state_tensor)
        
#         # Apply Boltzmann (softmax) with temperature
#         probs = F.softmax(logits / self.current_temp, dim=1)
        
#         # Sample action from the distribution
#         action_dist = torch.distributions.Categorical(probs)
#         action = action_dist.sample()
        
#         # Store log probability for training
#         log_prob = action_dist.log_prob(action)
        
#         return action.item(), log_prob
    
#     def store_transition(self, state, action, reward, next_state, done, log_prob):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.next_states.append(next_state)
#         self.dones.append(done)
#         self.log_probs.append(log_prob)
    
#     def calculate_returns(self):
#         returns = []
#         R = 0
        
#         # Calculate returns, going backwards from the end of the episode
#         for r, done in zip(reversed(self.rewards), reversed(self.dones)):
#             R = 0 if done else r + self.gamma * R
#             returns.insert(0, R)
        
#         return torch.tensor(returns, dtype=torch.float32).to(device)
    
#     def update_reinforce(self):
#         # Calculate returns
#         returns = self.calculate_returns()
        
#         # Normalize returns for stable training
#         returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
#         # Calculate loss
#         policy_loss = []
#         for log_prob, R in zip(self.log_probs, returns):
#             policy_loss.append(-log_prob * R)
        
#         policy_loss = torch.cat(policy_loss).sum()
        
#         # Update policy network
#         self.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         self.actor_optimizer.step()
        
#         # Reset episode data
#         self.reset_episode_data()
        
#         return policy_loss.item()
    
#     def update_actor_critic(self):
#         states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
#         next_states = torch.tensor(np.array(self.next_states), dtype=torch.float32).to(device)
#         rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
#         dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        
#         # Calculate advantage estimates
#         with torch.no_grad():
#             next_values = self.critic(next_states).squeeze()
            
#         # Calculate target values
#         target_values = rewards + self.gamma * next_values * (1 - dones)
        
#         # Get current values
#         current_values = self.critic(states).squeeze()
        
#         # Calculate advantage
#         advantages = target_values - current_values
        
#         # Value loss
#         value_loss = F.mse_loss(current_values, target_values.detach())
        
#         # Policy loss
#         policy_loss = []
#         for log_prob, advantage in zip(self.log_probs, advantages):
#             policy_loss.append(-log_prob * advantage.detach())
        
#         policy_loss = torch.cat(policy_loss).sum()
        
#         # Update critic network
#         self.critic_optimizer.zero_grad()
#         value_loss.backward()
#         self.critic_optimizer.step()
        
#         # Update actor network
#         self.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         self.actor_optimizer.step()
        
#         # Reset episode data
#         self.reset_episode_data()
        
#         return policy_loss.item(), value_loss.item()
    
#     def update(self):
#         if self.algorithm == "REINFORCE":
#             return self.update_reinforce()
#         elif self.algorithm == "ActorCritic":
#             return self.update_actor_critic()

#############################################
# COMMON FUNCTIONS
#############################################

def preprocess_state(state, env_name):
    """Preprocess the state based on the environment"""
    if 'ram' in env_name.lower():
        # Normalize RAM observations to [0,1]
        return state / 255.0
    else:
        # For other environments, no special preprocessing
        return state

def run_value_based_experiment(env_name, algorithm, use_replay_buffer, epsilon, learning_rate, seed):
    """Run a single experiment with value-based methods"""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    
    # Get observation and action dimensions
    if isinstance(env.observation_space, gym.spaces.Box):
        n_observations = env.observation_space.shape[0]
    else:
        n_observations = env.observation_space.n
        
    n_actions = env.action_space.n
    
    # Create agent
    agent = ValueBasedAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        lr=learning_rate,
        epsilon=epsilon,
        algorithm=algorithm,
        use_replay_buffer=use_replay_buffer
    )
    
    # Training loop
    episode_rewards = []
    for episode in range(100):  # 1000 episodes as per assignment
        state, _ = env.reset(seed=seed+episode)  # Different seed each episode
        state = preprocess_state(state, env_name)
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state, env_name)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        
    env.close()
    return episode_rewards

# def run_policy_based_experiment(env_name, algorithm, fixed_temp, initial_temp, final_temp, learning_rate, seed):
#     """Run a single experiment with policy-based methods"""
#     # Set random seeds for reproducibility
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
#     # Create environment
#     env = gym.make(env_name)
    
#     # Get observation and action dimensions
#     if isinstance(env.observation_space, gym.spaces.Box):
#         n_observations = env.observation_space.shape[0]
#     else:
#         n_observations = env.observation_space.n
        
#     n_actions = env.action_space.n
    
#     # Create agent
#     agent = PolicyBasedAgent(
#         n_observations=n_observations,
#         n_actions=n_actions,
#         lr=learning_rate,
#         algorithm=algorithm,
#         fixed_temp=fixed_temp,
#         initial_temp=initial_temp,
#         final_temp=final_temp
#     )
    
#     # Training loop
#     episode_rewards = []
#     total_episodes = 1000
    
#     for episode in range(total_episodes):
#         # Update temperature
#         agent.update_temperature(episode, total_episodes)
        
#         state, _ = env.reset(seed=seed+episode)
#         state = preprocess_state(state, env_name)
#         done = False
#         truncated = False
#         episode_reward = 0
        
#         while not (done or truncated):
#             action, log_prob = agent.select_action(state)
#             next_state, reward, done, truncated, _ = env.step(action)
#             next_state = preprocess_state(next_state, env_name)
            
#             agent.store_transition(state, action, reward, next_state, done, log_prob)
            
#             state = next_state
#             episode_reward += reward
        
#         # Update policy after episode is complete
#         agent.update()
#         episode_rewards.append(episode_reward)
        
#     env.close()
#     return episode_rewards

#############################################
# RUN EXPERIMENTS & PLOT RESULTS
#############################################

def run_all_value_based_experiments():
    """Run all experiments for value-based methods"""
    # Configuration parameters
    # environments = ["Acrobot-v1", "ALE/Assault-ram-v5"]
    environments = ["Acrobot-v1"]
    algorithms = ["DQN", "ExpectedSARSA"]
    # replay_buffer_options = [False, True]
    replay_buffer_options = [False]
    # epsilons = [1/4, 1/8, 1/16]
    # learning_rates = [0.01, 0.001, 0.0001]
    
    epsilons = [1/4]
    learning_rates = [0.001]
    
    seeds = range(1)
    
    # Dictionary to store all results
    all_results = {}
    
    # Progress tracking
    total_experiments = len(environments) * len(algorithms) * len(replay_buffer_options) * len(epsilons) * len(learning_rates) * len(seeds)
    experiment_count = 0
    
    start_time = time.time()
    
    for env_name in environments:
        all_results[env_name] = {}
        
        for use_buffer in replay_buffer_options:
            buffer_str = "with_buffer" if use_buffer else "no_buffer"
            all_results[env_name][buffer_str] = {}
            
            for algorithm in algorithms:
                all_results[env_name][buffer_str][algorithm] = {}
                
                for epsilon in epsilons:
                    for lr in learning_rates:
                        config_key = f"eps_{epsilon}_lr_{lr}"
                        all_results[env_name][buffer_str][algorithm][config_key] = []
                        
                        # Run experiments with different seeds
                        for seed in tqdm(seeds, desc=f"{env_name}, {algorithm}, {buffer_str}, eps={epsilon}, lr={lr}"):
                            experiment_count += 1
                            print(f"Running value-based experiment {experiment_count}/{total_experiments}")
                            
                            # Run the actual experiment
                            episode_rewards = run_value_based_experiment(
                                env_name=env_name,
                                algorithm=algorithm,
                                use_replay_buffer=use_buffer,
                                epsilon=epsilon,
                                learning_rate=lr,
                                seed=seed
                            )
                            
                            # Store results
                            all_results[env_name][buffer_str][algorithm][config_key].append(episode_rewards)
                            
                            # Save intermediate results
                            if experiment_count % 10 == 0:
                                with open("results/value_based_intermediate.json", "w") as f:
                                    json.dump(all_results, f)
    
    # Save final results
    with open("results/value_based_complete.json", "w") as f:
        json.dump(all_results, f)
    
    end_time = time.time()
    print(f"Value-based experiments completed in {(end_time - start_time) / 3600:.2f} hours")
    
    # Plot results
    plot_value_based_results(all_results)

# def run_all_policy_based_experiments():
#     """Run all experiments for policy-based methods"""
#     # Configuration parameters
#     environments = ["Acrobot-v1", "ALE/Assault-ram-v5"]
#     algorithms = ["REINFORCE", "ActorCritic"]
#     temperature_configs = [
#         {"fixed": True, "initial": 1.0, "final": 0.1},  # Fixed temperature
#         {"fixed": False, "initial": 1.0, "final": 0.1}   # Decreasing temperature
#     ]
#     learning_rate = 0.001  # Free to choose step size
#     seeds = range(10)  # 50 different seeds
    
#     # Dictionary to store all results
#     all_results = {}
    
#     # Progress tracking
#     total_experiments = len(environments) * len(algorithms) * len(temperature_configs) * len(seeds)
#     experiment_count = 0
    
#     start_time = time.time()
    
#     for env_name in environments:
#         all_results[env_name] = {}
        
#         for algorithm in algorithms:
#             all_results[env_name][algorithm] = {}
            
#             for temp_config in temperature_configs:
#                 config_key = "fixed_temp" if temp_config["fixed"] else "decreasing_temp"
#                 all_results[env_name][algorithm][config_key] = []
                
#                 # Run experiments with different seeds
#                 for seed in tqdm(seeds, desc=f"{env_name}, {algorithm}, {config_key}"):
#                     experiment_count += 1
#                     print(f"Running policy-based experiment {experiment_count}/{total_experiments}")
                    
#                     # Run the actual experiment
#                     episode_rewards = run_policy_based_experiment(
#                         env_name=env_name,
#                         algorithm=algorithm,
#                         fixed_temp=temp_config["fixed"],
#                         initial_temp=temp_config["initial"],
#                         final_temp=temp_config["final"],
#                         learning_rate=learning_rate,
#                         seed=seed
#                     )
                    
#                     # Store results
#                     all_results[env_name][algorithm][config_key].append(episode_rewards)
                    
#                     # Save intermediate results
#                     if experiment_count % 10 == 0:
#                         with open("results/policy_based_intermediate.json", "w") as f:
#                             json.dump(all_results, f)
    
#     # Save final results
#     with open("results/policy_based_complete.json", "w") as f:
#         json.dump(all_results, f)
    
#     end_time = time.time()
#     print(f"Policy-based experiments completed in {(end_time - start_time) / 3600:.2f} hours")
    
#     # Plot results
#     plot_policy_based_results(all_results)

def plot_value_based_results(all_results):
    """Plot the results for value-based methods"""
    # Set line styles for different hyperparameters
    
    line_styles = {
        "eps_0.25_lr_0.01": "-",
        "eps_0.25_lr_0.001": "--",
        "eps_0.25_lr_0.0001": "-.",
        "eps_0.125_lr_0.01": ":",
        "eps_0.125_lr_0.001": "-",
        "eps_0.125_lr_0.0001": "--",
        "eps_0.0625_lr_0.01": "-.",
        "eps_0.0625_lr_0.001": ":",
        "eps_0.0625_lr_0.0001": "-"
    }
    
    # Set colors for algorithms
    colors = {
        "DQN": "green",
        "ExpectedSARSA": "red"
    }
    
    # For each environment and buffer setting, create one plot with all hyperparameter configurations
    for env_name in all_results:
        for buffer_setting in all_results[env_name]:
            plt.figure(figsize=(12, 8))
            
            for algorithm in all_results[env_name][buffer_setting]:
                for config, results in all_results[env_name][buffer_setting][algorithm].items():
                    # Convert the list of lists to a numpy array
                    results_array = np.array(results)
                    
                    # Calculate mean and std across seeds
                    mean_rewards = np.mean(results_array, axis=0)
                    std_rewards = np.std(results_array, axis=0)
                    
                    # Plot mean with line style based on hyperparameters
                    x = np.arange(len(mean_rewards))
                    plt.plot(x, mean_rewards, 
                             label=f"{algorithm} {config}",
                             color=colors[algorithm],
                             linestyle=line_styles[config])
                    
                    # Add shaded region for standard deviation
                    plt.fill_between(x, 
                                    mean_rewards - std_rewards,
                                    mean_rewards + std_rewards,
                                    color=colors[algorithm],
                                    alpha=0.2)
            
            plt.title(f"{env_name} - {buffer_setting.replace('_', ' ')}")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plt.savefig(f"plots/{env_name}_{buffer_setting}_value_based.png")
            plt.close()

# def plot_policy_based_results(all_results):
#     """Plot the results for policy-based methods"""
#     # Set line styles for different temperature configurations
#     line_styles = {
#         "fixed_temp": "-",
#         "decreasing_temp": "--"
#     }
    
#     # Set colors for algorithms
#     colors = {
#         "REINFORCE": "green",
#         "ActorCritic": "red"
#     }
    
#     # For each environment, create one plot
#     for env_name in all_results:
#         plt.figure(figsize=(12, 8))
        
#         for algorithm in all_results[env_name]:
#             for config, results in all_results[env_name][algorithm].items():
#                 # Convert the list of lists to a numpy array
#                 results_array = np.array(results)
                
#                 # Calculate mean and std across seeds
#                 mean_rewards = np.mean(results_array, axis=0)
#                 std_rewards = np.std(results_array, axis=0)
                
#                 # Plot mean with line style based on temperature configuration
#                 x = np.arange(len(mean_rewards))
#                 plt.plot(x, mean_rewards, 
#                          label=f"{algorithm} {config}",
#                          color=colors[algorithm],
#                          linestyle=line_styles[config])
                
#                 # Add shaded region for standard deviation
#                 plt.fill_between(x, 
#                                 mean_rewards - std_rewards,
#                                 mean_rewards + std_rewards,
#                                 color=colors[algorithm],
#                                 alpha=0.2)
        
#         plt.title(f"{env_name} - Policy-based Methods")
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward")
#         plt.legend()
#         plt.grid(True)
        
#         # Save the plot
#         plt.savefig(f"plots/{env_name}_policy_based.png")
#         plt.close()

#############################################
# MAIN FUNCTION
#############################################

def main():
    """Main function to run all experiments"""
    print("Running value-based experiments...")
    run_all_value_based_experiments()

    print("Running policy-based experiments...")
    # run_all_policy_based_experiments()



if __name__ == "__main__":
    main()