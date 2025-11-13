import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json

# Constants
BATCH_SIZE = 64
GAMMA = 0.995
LEARNING_RATE = 0.0002
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQ = 500
LSTM_HIDDEN_SIZE = [256, 128, 64]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'action_features', 'reward', 'next_state', 'done'])

class DeterministicUserSimulator:
    """決定論的ルールベースのシミュレータ（ベースライン）"""
    def __init__(self, track_pool_size=1000):
        np.random.seed(42)
        self.track_preferences = np.random.rand(track_pool_size) * 0.4 + 0.6
        self.track_genres = np.random.randint(0, 10, track_pool_size)
        self.genre_preferences = None
        
    def reset_user_state(self):
        self.genre_preferences = np.random.rand(10) * 0.4 + 0.6
        
    def get_response(self, state, action, step):
        action_idx = int(action)
        base_response = self.track_preferences[action_idx]
        genre_match = self.genre_preferences[self.track_genres[action_idx]]
        base_response = base_response * 0.6 + genre_match * 0.4
        
        state_array = np.array(state)
        track_history = state_array[:len(state_array)//2]
        response_history = state_array[len(state_array)//2:]
        
        if step > 0:
            prev_responses = [r for r in response_history[:step] if r > 0]
            if len(prev_responses) > 0:
                recent_avg = np.mean(prev_responses[-3:])
                if abs(base_response - recent_avg) < 0.15:
                    base_response += 0.05
        
        if action_idx in track_history[:step]:
            base_response *= 0.3
        
        base_response += np.random.normal(0, 0.02)
        return np.clip(base_response, 0.0, 1.0)

class LSTMUserSimulator(nn.Module):
    """軽量版LSTMシミュレータ"""
    def __init__(self, input_size, hidden_sizes=LSTM_HIDDEN_SIZE):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[2], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)

def generate_expert_trajectories(deterministic_sim, n_trajectories=2000, session_length=20):
    """決定論的シミュレータで高品質な軌跡を生成"""
    print("=" * 60)
    print("高品質な学習データを生成中...")
    print("=" * 60)
    
    trajectories = []
    track_pool_size = 1000
    
    for traj_idx in range(n_trajectories):
        deterministic_sim.reset_user_state()
        
        track_history = []
        response_history = []
        
        for step in range(session_length):
            current_state = track_history + [0.0] * (20 - len(track_history))
            current_state += response_history + [0.0] * (20 - len(response_history))
            
            candidates = np.random.choice(track_pool_size, size=50, replace=False)
            best_action = None
            best_response = -1
            
            for candidate in candidates:
                if candidate not in track_history:
                    temp_response = deterministic_sim.get_response(
                        current_state, candidate, step
                    )
                    if temp_response > best_response:
                        best_response = temp_response
                        best_action = candidate
            
            if best_action is None:
                best_action = np.random.randint(0, track_pool_size)
                best_response = deterministic_sim.get_response(
                    current_state, best_action, step
                )
            
            state_with_action = current_state + [float(best_action)]
            trajectories.append((state_with_action, best_response))
            
            track_history.append(best_action)
            response_history.append(best_response)
        
        if (traj_idx + 1) % 500 == 0:
            avg_quality = np.mean([t[1] for t in trajectories[-10000:]])
            print(f"軌跡生成 {traj_idx + 1}/{n_trajectories}, 平均応答率: {avg_quality:.3f}")
    
    print("=" * 60)
    print(f"合計 {len(trajectories)} ステップの学習データを生成")
    print("=" * 60)
    return trajectories

def pretrain_lstm_with_expert_data(lstm_sim, trajectories, n_epochs=20, batch_size=64):
    """専門家データでLSTMを事前学習"""
    print("\nLSTMシミュレータの事前学習開始...")
    
    optimizer = optim.Adam(lstm_sim.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(n_epochs):
        random.shuffle(trajectories)
        epoch_losses = []
        
        for i in range(0, len(trajectories) - batch_size, batch_size):
            batch = trajectories[i:i+batch_size]
            states = torch.FloatTensor([t[0] for t in batch]).unsqueeze(1).to(DEVICE)
            targets = torch.FloatTensor([t[1] for t in batch]).unsqueeze(1).unsqueeze(2).to(DEVICE)
            
            outputs, _ = lstm_sim(states)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_sim.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    print(f"事前学習完了! 最終Loss: {avg_loss:.4f}\n")

class DuelingActionHeadDQN(nn.Module):
    """【優先度中】Dueling Architecture + Action Head DQN"""
    def __init__(self, state_dim, action_feature_dim):
        super(DuelingActionHeadDQN, self).__init__()
        
        # 共通の状態エンコーダ
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # アクション特徴エンコーダ
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Dueling: 状態価値V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Dueling: 行動優位性A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action_features):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_features = self.state_net(state)
        
        if action_features.dim() == 2:
            if action_features.size(0) == state.size(0):
                action_features = action_features.unsqueeze(1)
            else:
                action_features = action_features.unsqueeze(0)

        batch_size = state_features.size(0)
        if action_features.size(0) == 1 and batch_size > 1:
            action_features = action_features.expand(batch_size, -1, -1)
        
        action_features_processed = self.action_net(action_features.view(-1, action_features.size(-1)))
        action_features_processed = action_features_processed.view(batch_size, -1, 64)
        
        state_features_expanded = state_features.unsqueeze(1).expand(-1, action_features.size(1), -1)
        
        # 状態価値V(s)を計算 - 修正版
        value = self.value_stream(state_features)  # [batch_size, 1]
        # 正しく次元を調整
        num_actions = action_features.size(1)
        value = value.expand(-1, num_actions)  # [batch_size, num_actions]
        
        # 行動優位性A(s,a)を計算
        combined = torch.cat([state_features_expanded, action_features_processed], dim=-1)
        advantage = self.advantage_stream(combined.view(-1, combined.size(-1))).view(batch_size, -1)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, action_features, reward, next_state, done):
        self.buffer.append(Experience(state, action, action_features, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        action_features = torch.stack([e.action_features for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        return states, actions, action_features, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class MusicEnvironment:
    def __init__(self, user_simulator, track_pool_size=1000, session_length=20, state_dim=40):
        self.user_simulator = user_simulator
        self.track_pool_size = track_pool_size
        self.session_length = session_length
        self.state_dim = state_dim
        self.current_step = 0
        self.track_history = []
        self.response_history = []
        
    def reset(self):
        self.current_step = 0
        self.track_history = [0.0] * (self.state_dim // 2)
        self.response_history = [0.0] * (self.state_dim // 2)
        return self._get_state()

    def step(self, action):
        state = self._get_state()
        state_tensor = torch.FloatTensor([state + [float(action)]]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            response_prob, _ = self.user_simulator(state_tensor)
        response = float(response_prob.squeeze())
        
        self.track_history[self.current_step] = float(action)
        self.response_history[self.current_step] = response
        self.current_step += 1
        
        base_reward = response
        
        if float(action) in self.track_history[:self.current_step-1]:
            base_reward *= 0.2
        
        if self.current_step >= self.session_length:
            responses = np.array(self.response_history)
            avg_response = np.mean(responses)
            
            if avg_response > 0.9:
                base_reward += 2.0
            elif avg_response > 0.8:
                base_reward += 1.0
        
        done = self.current_step >= self.session_length
        return self._get_state(), base_reward, done

    def _get_state(self):
        return self.track_history + self.response_history

class DoubleDQNAgent:
    """【優先度中】Double DQN + Dueling DQN 実装"""
    def __init__(self, state_dim, action_feature_dim, use_dueling=True):
        self.use_dueling = use_dueling
        
        if use_dueling:
            self.policy_net = DuelingActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
            self.target_net = DuelingActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        else:
            from main_paperMD_hybrid import ActionHeadDQN
            self.policy_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
            self.target_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=30, min_lr=5e-5
        )
        
        self.memory = ReplayBuffer()
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state, action_features, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                q_values = self.policy_net(state_tensor, action_features)
                
                track_history = np.array(state)[:20]
                penalties = torch.zeros_like(q_values)
                for i in range(action_features.size(0)):
                    if i in track_history:
                        penalties[0, i] = -2.0
                
                return torch.argmax(q_values + penalties).item()
        else:
            return random.randrange(action_features.size(0))

    def train_step(self):
        """【優先度中】Double DQN学習ステップ"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        states, actions, action_features, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = states.to(DEVICE)
        action_features = action_features.to(DEVICE)
        rewards = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)
        
        # 現在のQ値を計算
        current_q = self.policy_net(states, action_features).squeeze(1)
        
        with torch.no_grad():
            # Double DQN: policy_netで次の行動を選択
            next_q_policy = self.policy_net(next_states, action_features)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            
            # target_netでその行動のQ値を評価
            next_q_target = self.target_net(next_states, action_features)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        return loss.item()
    
    def update_target_network(self):
        """エピソード単位でターゲットネットワークを更新"""
        self.episodes_done += 1
        if self.episodes_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  → ターゲットネットワーク更新 (Episode {self.episodes_done})")
    
    def save_model(self, save_dir, metrics):
        """【実装1】モデルと学習メトリクスを保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # モデルの重みを保存
        model_path = os.path.join(save_dir, 'policy_net.pth')
        torch.save(self.policy_net.state_dict(), model_path)
        
        # メトリクスをJSON形式で保存
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nモデルを保存: {model_path}")
        print(f"メトリクスを保存: {metrics_path}")
    
    def load_model(self, model_path):
        """保存したモデルを読み込み"""
        self.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f" モデルを読み込み: {model_path}")

def evaluate_agent(agent, env, cached_action_features, n_episodes=50):
    """【実装1】エージェントを評価（テストモード）"""
    print("\n" + "=" * 60)
    print("エージェント評価開始（テストセット）")
    print("=" * 60)
    
    total_rewards = []
    avg_responses = []
    duplicate_rates = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.session_length):
            # ε=0で完全に貪欲選択
            action = agent.select_action(state, cached_action_features, epsilon=0.0)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        # 統計を収集
        total_rewards.append(episode_reward)
        avg_responses.append(np.mean(env.response_history))
        
        # 重複率を計算
        unique_tracks = len(set([t for t in env.track_history if t > 0]))
        duplicate_rate = 1.0 - (unique_tracks / env.session_length)
        duplicate_rates.append(duplicate_rate)
    
    # 結果をまとめる
    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_response': np.mean(avg_responses),
        'std_response': np.std(avg_responses),
        'avg_duplicate_rate': np.mean(duplicate_rates),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }
    
    print(f"\n【評価結果】({n_episodes}エピソード)")
    print(f"  平均報酬: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均応答率: {results['avg_response']:.3f} ± {results['std_response']:.3f}")
    print(f"  重複率: {results['avg_duplicate_rate']:.1%}")
    print(f"  報酬範囲: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("=" * 60 + "\n")
    
    return results

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_double_dueling_{timestamp}'
    save_dir = f'saved_models/{timestamp}'
    writer = SummaryWriter(log_dir)
    
    print("\n" + "=" * 60)
    print("【完全版】Double DQN + Dueling DQN + モデル保存・評価")
    print("=" * 60)
    print("実装内容:")
    print("  1. Double DQN (Q値過大評価を防止)")
    print("  2. Dueling DQN (状態価値と行動優位性を分離)")
    print("  3. モデル保存・読み込み機能")
    print("  4. テストセットでの評価機能")
    print("=" * 60 + "\n")
    
    state_dim = 40
    action_feature_dim = 64
    track_pool_size = 1000
    session_length = 20
    
    # Step 1: 決定論的シミュレータで高品質データ生成
    det_sim = DeterministicUserSimulator(track_pool_size)
    trajectories = generate_expert_trajectories(det_sim, n_trajectories=2000)
    
    # Step 2: LSTMを事前学習
    lstm_sim = LSTMUserSimulator(state_dim + 1).to(DEVICE)
    pretrain_lstm_with_expert_data(lstm_sim, trajectories, n_epochs=20)
    
    # Step 3: Double DQN + Dueling DQN学習
    env = MusicEnvironment(lstm_sim, track_pool_size, session_length, state_dim)
    agent = DoubleDQNAgent(state_dim, action_feature_dim, use_dueling=True)
    
    n_episodes = 300
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.998
    
    best_avg_reward = float('-inf')
    rewards_history = []
    cached_action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    print("\n" + "=" * 60)
    print("Double DQN + Dueling DQN学習開始")
    print("=" * 60 + "\n")
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward, total_loss = 0, 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        for step in range(session_length):
            action = agent.select_action(state, cached_action_features, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, cached_action_features[action].cpu(), reward, next_state, done)
            
            loss = agent.train_step()
            total_loss += loss
            total_reward += reward
            state = next_state
        
        agent.update_target_network()
        
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-10:])
        agent.scheduler.step(avg_reward)
        
        avg_resp = np.mean(env.response_history)
        
        writer.add_scalar('Episode/Total_Reward', total_reward, episode)
        writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Episode/Avg_Response', avg_resp, episode)
        writer.add_scalar('Episode/Epsilon', epsilon, episode)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:3d} | Total: {total_reward:6.2f} | "
                  f"Avg: {avg_reward:6.2f} | Resp: {avg_resp:.3f} | ε: {epsilon:.3f}")
    
    writer.close()
    
    # 【実装1】学習完了後の評価とモデル保存
    print(f"\n学習完了! 最良平均報酬: {best_avg_reward:.2f}")
    
    # テストセットで評価
    test_results = evaluate_agent(agent, env, cached_action_features, n_episodes=50)
    
    # 学習メトリクスをまとめる
    training_metrics = {
        'timestamp': timestamp,
        'architecture': 'Double DQN + Dueling DQN',
        'n_episodes': n_episodes,
        'best_avg_reward': float(best_avg_reward),
        'final_avg_reward': float(avg_reward),
        'final_avg_response': float(avg_resp),
        'test_results': test_results,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'gamma': GAMMA,
            'learning_rate': LEARNING_RATE,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'update_target_freq': UPDATE_TARGET_FREQ
        }
    }
    
    # モデルとメトリクスを保存
    agent.save_model(save_dir, training_metrics)
    
    print(f"\n すべての処理が完了しました！")
    print(f" 保存ディレクトリ: {save_dir}")

if __name__ == "__main__":
    main()