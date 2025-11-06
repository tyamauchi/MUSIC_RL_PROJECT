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
    """
    LSTMベースのユーザーシミュレータ
    ユーザーの聴取履歴から次の状態と報酬を予測
    """
    def __init__(self, track_embed_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM: 曲の埋め込みを時系列処理
        self.lstm = nn.LSTM(
            track_embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # 報酬予測ヘッド (完聴/スキップ確率)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [完聴確率, スキップ確率, 離脱確率]
        )
        
        # 次の状態埋め込み
        self.state_head = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, history_embeds, hidden=None):
        """
        Args:
            history_embeds: (batch, seq_len, embed_dim) - 聴取履歴の曲埋め込み
            hidden: LSTM隠れ状態
        Returns:
            state: 次の状態表現
            reward_probs: [完聴, スキップ, 離脱]の確率
            hidden: 更新された隠れ状態
        """
        lstm_out, hidden = self.lstm(history_embeds, hidden)
        last_output = lstm_out[:, -1, :]  # 最後のタイムステップ
        
        # 報酬予測 (ユーザー反応)
        reward_probs = torch.softmax(self.reward_head(last_output), dim=-1)
        
        # 次の状態
        state = self.state_head(last_output)
        
        return state, reward_probs, hidden
    
    def init_hidden(self, batch_size=1):
        """LSTM隠れ状態の初期化"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)


# ========================================
# 2. Action-Head DQN
# ========================================
class ActionHeadDQN(nn.Module):
    """
    Action-Head DQN: (状態, 行動)ペアを入力してQ値を出力
    通常のDQNとは異なり、1つの行動に対するQ値のみを計算
    """
    def __init__(self, state_dim=256, action_dim=128, hidden_dims=[512, 256]):
        super(ActionHeadDQN, self).__init__()
        
        # 状態と行動を結合して処理
        input_dim = state_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        # Q値出力 (スカラー)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        Args:
            state: (batch, state_dim) - 現在の状態
            action: (batch, action_dim) - 候補曲の埋め込み
        Returns:
            q_value: (batch, 1) - その行動のQ値
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# ========================================
# 3. Experience Replay Buffer
# ========================================
class ReplayBuffer:
    """経験再生バッファ"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ========================================
# 4. Simulated Music Environment
# ========================================
class MusicEnvironment:
    """
    LSTMシミュレータを使った音楽推薦環境
    """
    def __init__(self, user_simulator, track_catalog, track_embed_dim=128):
        self.simulator = user_simulator
        self.track_catalog = track_catalog  # 全候補曲の埋め込み
        self.track_embed_dim = track_embed_dim
        self.reset()
    
    def reset(self):
        """エピソードをリセット"""
        self.history = []
        self.hidden = self.simulator.init_hidden()
        self.done = False
        self.step_count = 0
        
        # 初期状態 (ランダムな初期ユーザー状態)
        self.state = torch.randn(1, 256)
        return self.state.numpy()[0]
    
    def step(self, action_embed):
        """
        Args:
            action_embed: 選択された曲の埋め込み
        Returns:
            next_state, reward, done, info
        """
        self.step_count += 1
        
        # 履歴に追加
        self.history.append(action_embed)
        
        # 履歴を整形 (最新10曲)
        recent_history = self.history[-10:]
        history_tensor = torch.FloatTensor(recent_history).unsqueeze(0)
        
        # LSTMで次の状態と報酬を予測
        with torch.no_grad():
            next_state, reward_probs, self.hidden = self.simulator(
                history_tensor, self.hidden
            )
        
        # 報酬の決定 (確率的サンプリング)
        # [完聴: +1, スキップ: -0.5, 離脱: -1]
        action_outcome = torch.multinomial(reward_probs[0], 1).item()
        
        if action_outcome == 0:  # 完聴
            reward = 1.0
        elif action_outcome == 1:  # スキップ
            reward = -0.5
        else:  # 離脱
            reward = -1.0
            self.done = True
        
        # プレイリストの長さ制限
        if self.step_count >= 30:
            self.done = True
        
        self.state = next_state
        
        info = {
            'outcome': ['complete', 'skip', 'exit'][action_outcome],
            'step': self.step_count
        }
        
        return next_state.numpy()[0], reward, self.done, info


# ========================================
# 5. DQN Agent
# ========================================
class DQNAgent:
    """
    Action-Head DQNを使ったプレイリスト生成エージェント
    """
    def __init__(self, q_network, target_network, track_catalog, 
                 lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(q_network.state_dict())
        
        self.optimizer = optim.Adam(q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.track_catalog = track_catalog  # (num_tracks, embed_dim)
        self.replay_buffer = ReplayBuffer(capacity=50000)
    
    def select_action(self, state):
        """
        ε-greedy戦略で曲を選択
        全候補曲に対してQ値を計算し、最大のものを選択
        """
        if random.random() < self.epsilon:
            # 探索: ランダムに曲を選択
            idx = random.randint(0, len(self.track_catalog) - 1)
            return self.track_catalog[idx], idx
        
        # 活用: Q値が最大の曲を選択
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        best_q = float('-inf')
        best_action = None
        best_idx = 0
        
        with torch.no_grad():
            for idx, track_embed in enumerate(self.track_catalog):
                action_tensor = torch.FloatTensor(track_embed).unsqueeze(0)
                q_value = self.q_network(state_tensor, action_tensor).item()
                
                if q_value > best_q:
                    best_q = q_value
                    best_action = track_embed
                    best_idx = idx
        
        return best_action, best_idx
    
    def train(self, batch_size=64):
        """経験再生でQ-networkを更新"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 現在のQ値
        current_q = self.q_network(states, actions).squeeze()
        
        # 次の状態での最大Q値を計算 (全候補曲を評価)
        with torch.no_grad():
            next_q_values = []
            for i in range(batch_size):
                max_next_q = float('-inf')
                for track_embed in self.track_catalog[:100]:  # 計算量削減のため一部のみ
                    action_tensor = torch.FloatTensor(track_embed).unsqueeze(0)
                    next_q = self.target_network(
                        next_states[i:i+1], action_tensor
                    ).item()
                    max_next_q = max(max_next_q, next_q)
                next_q_values.append(max_next_q)
            
            next_q_values = torch.FloatTensor(next_q_values)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss計算とバックプロパゲーション
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Target networkを更新"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """探索率を減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ========================================
# 6. Training Loop
# ========================================
def train_playlist_agent(num_episodes=1000, max_steps=30):
    """
    DQNエージェントをLSTM環境で訓練
    """
    # モデルの初期化
    track_embed_dim = 128
    state_dim = 256
    
    # 仮の曲カタログ (実際にはSpotifyの曲埋め込みを使用)
    num_tracks = 1000
    track_catalog = np.random.randn(num_tracks, track_embed_dim)
    
    # LSTMユーザーシミュレータ
    user_simulator = LSTMUserSimulator(track_embed_dim, hidden_dim=state_dim)
    
    # 環境
    env = MusicEnvironment(user_simulator, track_catalog, track_embed_dim)
    
    # DQNエージェント
    q_network = ActionHeadDQN(state_dim, track_embed_dim)
    target_network = ActionHeadDQN(state_dim, track_embed_dim)
    agent = DQNAgent(q_network, target_network, track_catalog)
    
    # 訓練ループ
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 行動選択
            action_embed, action_idx = agent.select_action(state)
            
            # 環境とのインタラクション
            next_state, reward, done, info = env.step(action_embed)
            episode_reward += reward
            
            # 経験をバッファに保存
            agent.replay_buffer.push(state, action_embed, reward, next_state, done)
            
            # Q-networkの訓練
            loss = agent.train(batch_size=64)
            
            state = next_state
            
            if done:
                break
        
        # Target networkの更新
        if episode % 10 == 0:
            agent.update_target_network()
        
        # εの減衰
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        
        # ログ出力
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, episode_rewards


# ========================================
# 7. 実行例
# ========================================
if __name__ == "__main__":
    print("Training DQN Agent for Music Playlist Generation...")
    print("=" * 60)
    
    # 訓練実行
    trained_agent, rewards = train_playlist_agent(num_episodes=500, max_steps=30)
    
    print("\nTraining completed!")
    print(f"Final average reward: {np.mean(rewards[-50:]):.2f}")
    
    # プレイリスト生成のデモ
    print("\n" + "=" * 60)
    print("Generating a sample playlist...")
    
    # 環境の再作成
    track_catalog = np.random.randn(1000, 128)
    user_simulator = LSTMUserSimulator(128, 256)
    env = MusicEnvironment(user_simulator, track_catalog, 128)
    
    state = env.reset()
    playlist = []
    
    for i in range(10):
        action_embed, action_idx = trained_agent.select_action(state)
        playlist.append(action_idx)
        next_state, reward, done, info = env.step(action_embed)
        print(f"  Track {i+1}: idx={action_idx}, outcome={info['outcome']}, reward={reward:.2f}")
        state = next_state
        if done:
            break
    
    print(f"\nGenerated playlist with {len(playlist)} tracks")
          
