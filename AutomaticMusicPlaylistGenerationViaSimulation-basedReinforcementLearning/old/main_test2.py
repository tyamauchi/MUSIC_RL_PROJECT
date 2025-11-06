import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ========================================
# 1. LSTM-based Environment Model (User Simulator)
# ========================================
class LSTMUserSimulator(nn.Module):
    """LSTMベースのユーザーシミュレータ"""
    def __init__(self, track_embed_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            track_embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [完聴, スキップ, 離脱]
        )
        self.state_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, history_embeds, hidden=None):
        lstm_out, hidden = self.lstm(history_embeds, hidden)
        last_output = lstm_out[:, -1, :]
        reward_probs = torch.softmax(self.reward_head(last_output), dim=-1)
        state = self.state_head(last_output)
        return state, reward_probs, hidden

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)

# ========================================
# 2. Dueling DQN (state & action concatenated -> value + advantage)
# ========================================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim=256, action_dim=128, hidden_dims=[512, 256]):
        super(DuelingDQN, self).__init__()
        input_dim = state_dim + action_dim
        # shared feature extractor
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = h_dim
        self.feature = nn.Sequential(*layers)

        # value and advantage heads
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1)
        )

    def forward(self, state, action):
        """
        state: (batch, state_dim)
        action: (batch, action_dim)
        return: (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        feat = self.feature(x)
        value = self.value_head(feat)  # (batch, 1)
        adv = self.adv_head(feat)      # (batch, 1)
        # Q = V + (A - mean(A))
        q = value + (adv - adv.mean(dim=0, keepdim=True))
        return q  # shape (batch, 1)

# ========================================
# 3. Replay Buffer
# ========================================
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    def __len__(self):
        return len(self.buffer)

# ========================================
# 4. Music Environment
# ========================================
class MusicEnvironment:
    def __init__(self, user_simulator, track_catalog, track_embed_dim=128):
        self.simulator = user_simulator
        self.track_catalog = track_catalog
        self.track_embed_dim = track_embed_dim
        self.reset()

    def reset(self):
        self.history = []
        self.hidden = self.simulator.init_hidden()
        self.done = False
        self.step_count = 0
        # initial state: small random vector (matches simulator hidden_dim)
        self.state = torch.randn(1, self.simulator.hidden_dim)
        return self.state.numpy()[0]

    def step(self, action_embed):
        self.step_count += 1
        self.history.append(action_embed)
        recent_history = self.history[-10:]
        history_tensor = torch.FloatTensor(recent_history).unsqueeze(0)  # (1, seq, embed)

        with torch.no_grad():
            next_state, reward_probs, self.hidden = self.simulator(history_tensor, self.hidden)

        action_outcome = torch.multinomial(reward_probs[0], 1).item()

        if action_outcome == 0:
            reward = 1.0
        elif action_outcome == 1:
            reward = -0.2
        else:
            reward = -1.0
            self.done = True

        if self.step_count >= 30:
            self.done = True

        self.state = next_state
        info = {'outcome': ['complete', 'skip', 'exit'][action_outcome], 'step': self.step_count}
        return next_state.numpy()[0], reward, self.done, info

# ========================================
# 5. DQN Agent (Double DQN + soft target update + dueling)
# ========================================
class DQNAgent:
    def __init__(self, q_network, target_network, track_catalog,
                 lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, tau=0.005, device=None):
        self.device = device or torch.device("cpu")
        self.q_network = q_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.target_network.load_state_dict(q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        self.track_catalog = [torch.FloatTensor(te).to(self.device) for te in track_catalog]
        self.replay_buffer = ReplayBuffer(capacity=20000)
        self.tau = tau

    def select_action(self, state):
        # state: numpy array (state_dim,)
        if random.random() < self.epsilon:
            idx = random.randint(0, len(self.track_catalog) - 1)
            return self.track_catalog[idx].cpu().numpy(), idx

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        best_q = float('-inf')
        best_action = None
        best_idx = 0
        with torch.no_grad():
            for idx, track_embed in enumerate(self.track_catalog):
                action_tensor = track_embed.unsqueeze(0)  # (1, action_dim)
                q_value = self.q_network(state_tensor, action_tensor).item()
                if q_value > best_q:
                    best_q, best_action, best_idx = q_value, track_embed, idx
        return best_action.cpu().numpy(), best_idx

    def train(self, batch_size=128, candidate_subset_size=200):
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # current Q
        current_q = self.q_network(states, actions).squeeze(1)  # (batch,)

        # Double DQN: use online q_network to select next action, use target_network to evaluate
        next_q_values = []
        # To limit computation, use only a subset of catalog if it's huge
        catalog_size = len(self.track_catalog)
        use_n = min(candidate_subset_size, catalog_size)

        with torch.no_grad():
            for i in range(batch_size):
                # sample a subset of candidate actions uniformly for speed
                if use_n < catalog_size:
                    candidate_indices = random.sample(range(catalog_size), use_n)
                else:
                    candidate_indices = range(catalog_size)

                # compute q_online for next state for candidates -> select argmax
                best_idx = None
                best_q_online = float('-inf')
                for idx in candidate_indices:
                    act = self.track_catalog[idx].unsqueeze(0)  # (1, action_dim)
                    q_online = self.q_network(next_states[i:i+1], act).item()
                    if q_online > best_q_online:
                        best_q_online = q_online
                        best_idx = idx

                # evaluate with target network
                chosen_act = self.track_catalog[best_idx].unsqueeze(0)  # (1, action_dim)
                q_target = self.target_network(next_states[i:i+1], chosen_act).item()
                next_q_values.append(q_target)

            next_q_values = torch.FloatTensor(next_q_values).to(self.device).unsqueeze(1)  # (batch,1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values  # (batch,1)
            target_q = target_q.squeeze(1)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        # soft update: target = tau * online + (1-tau) * target
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_( (1.0 - self.tau) * target_param.data + self.tau * param.data )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========================================
# 6. Training Loop
# ========================================
def train_playlist_agent(num_episodes=1000, max_steps=30):
    track_embed_dim, state_dim = 128, 256

    # --- ジャンルクラスタ構造を持つ曲埋め込み ---
    cluster_centers = np.random.randn(3, track_embed_dim)
    track_catalog = np.vstack([
        cluster_centers[i] + 0.1 * np.random.randn(333, track_embed_dim)
        for i in range(3)
    ])
    track_catalog = track_catalog[:1000]

    user_simulator = LSTMUserSimulator(track_embed_dim, hidden_dim=state_dim)
    env = MusicEnvironment(user_simulator, track_catalog, track_embed_dim)
    q_network = DuelingDQN(state_dim, track_embed_dim)
    target_network = DuelingDQN(state_dim, track_embed_dim)
    agent = DQNAgent(q_network, target_network, track_catalog,
                     lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, tau=0.005)

    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        state, episode_reward = env.reset(), 0
        for step in range(max_steps):
            action_embed, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action_embed)
            agent.replay_buffer.push(state, action_embed, reward, next_state, float(done))
            # training step
            loss = agent.train(batch_size=128)
            state = next_state
            episode_reward += reward
            if done:
                break

        # soft update every episode (you can adjust frequency)
        agent.update_target_network()
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        if episode % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            print(f"Ep {episode}/{num_episodes} | Avg Reward(last50): {avg_r:.3f} | Eps: {agent.epsilon:.3f}")

    return agent, episode_rewards

# ========================================
# 7. 実行例
# ========================================
if __name__ == "__main__":
    print("Training improved DQN Agent (Dueling + Double DQN + Soft target update)...")
    print("=" * 80)
    trained_agent, rewards = train_playlist_agent(num_episodes=500, max_steps=30)
    print(f"\nTraining completed! Final avg reward (last50): {np.mean(rewards[-50:]):.3f}")
