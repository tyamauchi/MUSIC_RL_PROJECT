import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np  # ensure numpy is imported for array operations

# Constants
BATCH_SIZE = 64  # バッチサイズを適度に増加
INITIAL_MEMORY = 2000  # より多くの経験を蓄積
GAMMA = 0.995  # 将来の報酬をより重視
LEARNING_RATE = 0.0002  # 学習率を少し上げる
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQ = 5  # より頻繁なターゲットネットワークの更新
LSTM_HIDDEN_SIZE = [500, 200, 200]  # 3-layer LSTM as per paper

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'action_features', 'reward', 'next_state', 'done'])

class LSTMUserSimulator(nn.Module):
    """Sequential World Model (SWM) for user behavior simulation"""
    def __init__(self, input_size, hidden_sizes=LSTM_HIDDEN_SIZE):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        # 3-layer LSTM as described in the paper
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        # Output layer for completion probability
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[2], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        # Forward pass through LSTM layers
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        # Predict completion probability
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)

class ActionHeadDQN(nn.Module):
    """Action Head DQN for playlist generation"""
    def __init__(self, state_dim, action_feature_dim):
        super(ActionHeadDQN, self).__init__()
        
        # State processing
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action processing
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Q-value output
        self.q_net = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action_features):
        # Process state features - expand state if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_features = self.state_net(state)
        
        # action_features can be provided in two useful shapes:
        #  - [num_actions, feat]  : a global candidate set (used at inference)
        #  - [batch, feat]        : one selected action feature per sample (used for replay training)
        # Convert to a unified shape [batch, num_actions, feat] for processing.
        if action_features.dim() == 2:
            if action_features.size(0) == state.size(0):
                # per-sample single action feature -> [batch, 1, feat]
                action_features = action_features.unsqueeze(1)
            else:
                # global candidate set -> [1, num_actions, feat]
                action_features = action_features.unsqueeze(0)

        batch_size = state_features.size(0)
        if action_features.size(0) == 1 and batch_size > 1:
            # Expand global candidate set across the batch
            action_features = action_features.expand(batch_size, -1, -1)
        
        # Process action features
        action_features_processed = self.action_net(action_features.view(-1, action_features.size(-1)))
        action_features_processed = action_features_processed.view(batch_size, -1, 64)
        
        # Expand state features to match action features
        state_features_expanded = state_features.unsqueeze(1).expand(-1, action_features.size(1), -1)
        
        # Combine state and action features
        combined = torch.cat([state_features_expanded, action_features_processed], dim=-1)
        
        # Compute Q-values for all state-action pairs
        q_values = self.q_net(combined.view(-1, combined.size(-1))).view(batch_size, -1)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
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
    """Simulation environment for playlist generation"""
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
        self.track_history = [0.0] * (self.state_dim // 2)  # Initialize with zeros
        self.response_history = [0.0] * (self.state_dim // 2)  # Initialize with zeros
        return self._get_state()

    def step(self, action):
        # Simulate user response using the SWM
        state = self._get_state()
        state_tensor = torch.FloatTensor([state + [float(action)]]).unsqueeze(0)
        response_prob, _ = self.user_simulator(state_tensor)
        response = float(response_prob.squeeze())
        
        # Update histories
        self.track_history[self.current_step] = float(action)
        self.response_history[self.current_step] = response
        self.current_step += 1
        
        # 基本報酬の計算（対数スケーリング）
        base_reward = 10 * np.log1p(response * 2)  # より穏やかな報酬スケーリング
        
        # 連続応答に基づく報酬調整
        if self.current_step > 1:
            prev_response = self.response_history[self.current_step-2]
            curr_bonus = 0.0
            
            # 段階的な連続応答ボーナス（より適度な差別化）
            if prev_response > 0.9 and response > 0.9:
                curr_bonus = 2.0  # 高品質な連続応答に対するボーナス
                if prev_response > 0.95 and response > 0.95:
                    curr_bonus = 3.0  # 最高品質への追加ボーナス
            elif prev_response > 0.8 and response > 0.8:
                curr_bonus = 1.5
            elif prev_response > 0.7 and response > 0.7:
                curr_bonus = 1.0
                
            # 改善に対するボーナス
            if response > prev_response * 1.1:  # 10%以上の改善
                curr_bonus += 0.5
            
            # 応答品質の低下に対するペナルティ
            if response < prev_response * 0.8:
                base_reward *= 0.7  # 急激な品質低下へのペナルティ
            
            base_reward *= (1.0 + curr_bonus)
        
        # ペナルティ：繰り返しトラックの場合
        if float(action) in self.track_history[:self.current_step-1]:
            base_reward *= 0.2  # より強いペナルティ
        
        # セッション終了時の包括的評価
        if self.current_step >= self.session_length:
            responses = np.array(self.response_history)
            avg_response = np.mean(responses)
            consistency = 1.0 - np.std(responses)  # 一貫性の評価
            
            # より積極的な改善評価
            recent_responses = responses[-5:]  # 直近5ステップの評価
            recent_improvement = np.mean(np.diff(recent_responses) > 0)  # 最近の改善傾向
            overall_improvement = np.mean(np.diff(responses) > 0)  # 全体的な改善傾向
            
            # 総合的な品質評価（より細かい段階分け）
            quality_bonus = 0.0
            if avg_response > 0.95 and consistency > 0.9:
                quality_bonus = 15.0  # 最高品質ボーナス
            elif avg_response > 0.9 and consistency > 0.85:
                quality_bonus = 12.0
            elif avg_response > 0.85 and consistency > 0.8:
                quality_bonus = 9.0
            elif avg_response > 0.8 and consistency > 0.75:
                quality_bonus = 6.0
            elif avg_response > 0.75:
                quality_bonus = 3.0
                
            # 改善傾向へのボーナス
            if recent_improvement > 0.6:  # 直近の改善が良好
                quality_bonus *= 1.5
            
            # 最終報酬の計算
            base_reward += quality_bonus
            base_reward *= (1.0 + consistency * 0.5 + recent_improvement * 0.3 + overall_improvement * 0.2)
        
        # Check if session is done
        done = self.current_step >= self.session_length
        
        return self._get_state(), base_reward, done

    def _get_state(self):
        # Combine track history and responses as state
        state = self.track_history + self.response_history
        return state

class DQNAgent:
    """DQN agent for learning playlist generation policy"""
    def __init__(self, state_dim, action_feature_dim):
        self.policy_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', 
                                                            factor=0.5, patience=5, 
                                                            verbose=True,
                                                            min_lr=1e-5)  # 最小学習率を設定
        self.memory = ReplayBuffer()
        self.steps_done = 0
        self.max_grad_norm = 1.0  # 勾配クリッピングの閾値

    def select_action(self, state, action_features, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.policy_net(state_tensor, action_features)
                
                # Add novelty bonus to encourage exploration
                state_array = np.array(state)
                track_history = state_array[:len(state_array)//2]
                action_penalties = torch.zeros_like(q_values)
                
                for i in range(action_features.size(0)):
                    if i in track_history:
                        action_penalties[0, i] = -0.5  # Penalize repeated tracks
                
                q_values += action_penalties
                return torch.argmax(q_values).item()
        else:
            return random.randrange(action_features.size(0))

    def train_step(self):
        # メモリが最小サイズに達していない場合はスキップ
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        # バッチサイズの動的調整
        if len(self.memory) < INITIAL_MEMORY:
            actual_batch_size = min(48, len(self.memory))  # より大きな初期バッチサイズ
        else:
            actual_batch_size = min(BATCH_SIZE, len(self.memory))
            
        # 追加の安定化措置
        if self.steps_done % 50 == 0:  # より頻繁な追加学習
            actual_batch_size = min(128, len(self.memory))
            
        # PERの模倣: 新しい経験を優先的に学習
        if len(self.memory) > actual_batch_size * 2:
            actual_batch_size = int(actual_batch_size * 1.5)
        
        # バッチのサンプリングと学習
        total_loss = 0.0
        n_updates = 1  # 1回の更新に制限
        
        for _ in range(n_updates):
            states, actions, action_features, rewards, next_states, dones = self.memory.sample(actual_batch_size)
        
        # Move tensors to device
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        action_features = action_features.to(DEVICE)
        rewards = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)
        
        # Get Q-values for current state-action pairs
        # Note: when using per-sample stored action_features, policy_net(states, action_features)
        # returns shape [batch, 1] (one q-value per sample). We therefore squeeze and use it directly.
        current_q = self.policy_net(states, action_features).squeeze(1)
        
        # Compute target Q values with Double DQN (using stored per-sample next action features)
        with torch.no_grad():
            # Evaluate next-q directly using the target network and the stored per-sample action features
            next_q = self.target_net(next_states, action_features).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        # Compute Huber loss (より安定した学習のため)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

def save_models(user_simulator, agent, save_path='saved_models'):
    """モデルの保存"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # User Simulatorの保存
    torch.save(user_simulator.state_dict(), f'{save_path}/user_simulator.pth')
    
    # DQNモデルの保存
    torch.save(agent.policy_net.state_dict(), f'{save_path}/dqn_model.pth')

def main():
    # Initialize parameters
    state_dim = 40  # Example: 20 tracks + 20 responses
    action_feature_dim = 64  # Example: track embedding dimension (reduced from 128)
    track_pool_size = 1000
    session_length = state_dim // 2  # Half for tracks, half for responses
    
    # モデルの初期化とGPUへの移動
    user_simulator = LSTMUserSimulator(state_dim + 1).to(DEVICE)  # +1 for action
    env = MusicEnvironment(user_simulator, track_pool_size, session_length, state_dim)
    agent = DQNAgent(state_dim, action_feature_dim)
    
    # Training parameters
    n_episodes = 300  # より現実的なエピソード数
    epsilon_start = 1.0
    epsilon_end = 0.2  # 最小探索率を増加
    epsilon_decay = 0.995  # より緩やかな減衰
    
    # 早期停止のパラメータ
    patience = 50  # より長い忍耐期間
    min_delta = 0.0005  # より小さな改善でも継続
    min_episodes = 100  # 最小学習エピソード数
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    # アクション特徴量をGPUに事前にキャッシュ
    cached_action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    # Store metrics for analysis
    rewards_history = []
    moving_avg_window = 10
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        while True:
            # キャッシュされたアクション特徴量を使用
            action_features = cached_action_features
            
            # Select and execute action with current epsilon value
            action = agent.select_action(state, action_features, epsilon)
            next_state, reward, done = env.step(action)
            
            # Store experience
            # Save only the selected action's feature vector (prevent storing entire candidate pool)
            try:
                selected_action_feat = action_features[action].cpu().detach()
            except Exception:
                # fallback: if action_features is batch-shaped with single action per sample
                selected_action_feat = action_features.cpu().detach()
            agent.memory.push(state, action, selected_action_feat, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            total_loss += loss
            total_reward += reward
            state = next_state
            
            if done:
                rewards_history.append(total_reward)
                avg_reward = np.mean(rewards_history[-moving_avg_window:])
                
                # Learning rateスケジューラーの更新
                agent.scheduler.step(avg_reward)
                
                # 早期停止の判定（最小エピソード数以降）
                if episode >= min_episodes:
                    if avg_reward > best_avg_reward + min_delta:
                        best_avg_reward = avg_reward
                        patience_counter = 0
                        # 新しいベストモデルを保存
                        save_models(user_simulator, agent, save_path='saved_models_best')
                    else:
                        patience_counter += 1
                elif avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                
                memory_size = len(agent.memory)
                print(f"Episode {episode + 1}, "
                      f"Total Reward: {total_reward:.4f}, "
                      f"Average Reward: {avg_reward:.4f}, "
                      f"Loss: {total_loss/session_length:.4f}, "
                      f"Epsilon: {epsilon:.4f}, "
                      f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}, "
                      f"Memory: {memory_size}/{MEMORY_SIZE}")

                # 定期的にモデルを保存（例：50エピソードごと）
                if (episode + 1) % 50 == 0:
                    save_models(user_simulator, agent)

                # 早期停止条件のチェック
                if patience_counter >= patience:
                    print(f"\n早期停止: {patience}エピソード連続で改善が見られませんでした")
                    print(f"最良の平均報酬: {best_avg_reward:.4f}")
                    save_models(user_simulator, agent, save_path='saved_models_best')
                    return

                break

if __name__ == "__main__":
    main()
