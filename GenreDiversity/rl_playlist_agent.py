# rl_playlist_agent_v2.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from spotify_dqn_env import SpotifyPlaylistEnv

# -------------------------
# Policy ネットワーク
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)

# -------------------------
# RL エージェント（REINFORCE）
# -------------------------
class RLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.log_probs = []
        self.rewards = []
        self.entropies = []
    
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        self.log_probs.append(m.log_prob(action))
        self.entropies.append(m.entropy())
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        for log_prob, R, entropy in zip(self.log_probs, returns, self.entropies):
            loss += - log_prob * R - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs, self.rewards, self.entropies = [], [], []

# -------------------------
# メイン学習ループ
# -------------------------
if __name__ == "__main__":
    df = pd.read_csv("songs_100.csv")
    features = ['danceability','energy','tempo','valence']
    song_features = df[features].values
    user_profile = song_features[0]
    genres = df['genre'].tolist()
    
    env = SpotifyPlaylistEnv(song_features, user_profile, genres)
    
    state_dim = user_profile.shape[0]
    action_dim = env.n_songs
    
    agent = RLAgent(state_dim, action_dim, lr=1e-4, gamma=0.99, entropy_coef=0.01)
    
    episodes = 1000
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # --- 多様性強化 ---
            # 過去に選ばれた曲のジャンル数が少ない場合は報酬を増やす
            unique_genres = len(set([genres[i] for i in env.played_songs] + [genres[action]]))
            diversity_bonus = 0.5 * (unique_genres / len(genres))
            reward += diversity_bonus
            
            agent.store_reward(reward)
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        agent.finish_episode()
        rewards_history.append(ep_reward)
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Total Reward: {ep_reward:.2f}")

    # 学習済みモデル保存
    torch.save(agent.policy.state_dict(), "rl_playlist_policy_v2.pth")
    print("✅ 学習済みモデルを保存しました")

    # 学習曲線を描画
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Playlist Learning Curve (Diversity Boosted)")
    plt.legend()
    plt.show()
