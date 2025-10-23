# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# DQNネットワーク
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# リプレイバッファ
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# DQN学習関数
def train_dqn(env, episodes=500, batch_size=32, gamma=0.99, lr=1e-3, 
              epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, verbose=True):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    replay = ReplayBuffer()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        while not done:
            step_count += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = net(torch.FloatTensor(state).to(device))
                    action = int(torch.argmax(q_vals).item())

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_vals = net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    q_next = net(next_states).max(1)[0]
                target = rewards + gamma * q_next * (1 - dones)
                loss = nn.MSELoss()(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep + 1) % 10 == 0:
            print(f"[Episode {ep+1}/{episodes}] Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {epsilon:.2f}, Steps: {step_count}")

    print("学習完了！")
    return net
