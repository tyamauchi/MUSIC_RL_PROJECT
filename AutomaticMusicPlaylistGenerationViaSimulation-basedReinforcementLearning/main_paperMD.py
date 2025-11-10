import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

# Constants
BATCH_SIZE = 64
INITIAL_MEMORY = 2000
GAMMA = 0.995
LEARNING_RATE = 0.0002
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQ = 5
LSTM_HIDDEN_SIZE = [500, 200, 200]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'action_features', 'reward', 'next_state', 'done'])

class LSTMUserSimulator(nn.Module):
    """Sequential World Model (SWM) for user behavior simulation"""
    def __init__(self, input_size, hidden_sizes=LSTM_HIDDEN_SIZE):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[2], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)

def pretrain_user_simulator(simulator, n_episodes=500, batch_size=32):
    """
    LSTMユーザーシミュレータを事前学習
    
    目的: 現実的なユーザー応答パターンを学習させる
    """
    print("=" * 60)
    print("LSTMユーザーシミュレータの事前学習を開始...")
    print("=" * 60)
    
    optimizer = optim.Adam(simulator.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 合成データ生成パラメータ
    track_pool_size = 1000
    track_qualities = np.random.rand(track_pool_size) * 0.4 + 0.6  # 0.6-1.0
    
    for episode in range(n_episodes):
        # セッションの長さ
        session_length = 20
        
        # シミュレートされたセッション
        states = []
        targets = []
        
        # ユーザーの好みを設定
        user_genre_pref = np.random.rand(10) * 0.4 + 0.6
        track_genres = np.random.randint(0, 10, track_pool_size)
        
        track_history = []
        response_history = []
        
        for step in range(session_length):
            # ランダムに曲を選択
            action = np.random.randint(0, track_pool_size)
            
            # 状態を構築
            current_state = track_history + [0.0] * (20 - len(track_history))
            current_state += response_history + [0.0] * (20 - len(response_history))
            current_state.append(float(action))
            
            # ターゲット応答を計算（ルールベース）
            base_response = track_qualities[action]
            genre_match = user_genre_pref[track_genres[action]]
            target_response = base_response * 0.6 + genre_match * 0.4
            
            # 連続性ボーナス
            if len(response_history) > 0:
                prev_avg = np.mean(response_history[-3:])
                if abs(target_response - prev_avg) < 0.15:
                    target_response += 0.05
            
            # 重複ペナルティ
            if action in track_history:
                target_response *= 0.3
            
            # ノイズ追加
            target_response += np.random.normal(0, 0.02)
            target_response = np.clip(target_response, 0.0, 1.0)
            
            states.append(current_state)
            targets.append(target_response)
            
            track_history.append(action)
            response_history.append(target_response)
        
        # バッチ学習
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            if len(batch_states) < 2:
                continue
            
            state_tensor = torch.FloatTensor(batch_states).unsqueeze(1).to(DEVICE)
            target_tensor = torch.FloatTensor(batch_targets).unsqueeze(1).unsqueeze(2).to(DEVICE)
            
            outputs, _ = simulator(state_tensor)
            loss = criterion(outputs, target_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simulator.parameters(), 1.0)
            optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"事前学習 Episode {episode + 1}/{n_episodes}, Loss: {loss.item():.4f}")
    
    print("=" * 60)
    print(f"事前学習完了! 最終Loss: {loss.item():.4f}")
    print("=" * 60)
    print()

class ActionHeadDQN(nn.Module):
    """Action Head DQN for playlist generation"""
    def __init__(self, state_dim, action_feature_dim):
        super(ActionHeadDQN, self).__init__()
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.q_net = nn.Sequential(
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
        
        combined = torch.cat([state_features_expanded, action_features_processed], dim=-1)
        
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
        self.user_mood = np.random.uniform(0.6, 1.0)
        self.user_pickiness = np.random.uniform(0.0, 0.3)
        
    def reset(self):
        self.current_step = 0
        self.track_history = [0.0] * (self.state_dim // 2)
        self.response_history = [0.0] * (self.state_dim // 2)
        self.user_mood = np.random.uniform(0.6, 1.0)
        self.user_pickiness = np.random.uniform(0.0, 0.3)
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
        
        # 報酬計算
        base_reward = 10 * np.log1p(response * 2)
        
        if self.current_step > 1:
            prev_response = self.response_history[self.current_step-2]
            curr_bonus = 0.0
            
            if prev_response > 0.9 and response > 0.9:
                curr_bonus = 2.0
                if prev_response > 0.95 and response > 0.95:
                    curr_bonus = 3.0
            elif prev_response > 0.8 and response > 0.8:
                curr_bonus = 1.5
            elif prev_response > 0.7 and response > 0.7:
                curr_bonus = 1.0
                
            if response > prev_response * 1.1:
                curr_bonus += 0.5
            
            if response < prev_response * 0.8:
                base_reward *= 0.7
            
            base_reward *= (1.0 + curr_bonus)
        
        if float(action) in self.track_history[:self.current_step-1]:
            base_reward *= 0.2
        
        if self.current_step >= self.session_length:
            responses = np.array(self.response_history)
            avg_response = np.mean(responses)
            consistency = 1.0 - np.std(responses)
            
            recent_responses = responses[-5:]
            recent_improvement = np.mean(np.diff(recent_responses) > 0)
            overall_improvement = np.mean(np.diff(responses) > 0)
            
            quality_bonus = 0.0
            if avg_response > 0.95 and consistency > 0.9:
                quality_bonus = 15.0
            elif avg_response > 0.9 and consistency > 0.85:
                quality_bonus = 12.0
            elif avg_response > 0.85 and consistency > 0.8:
                quality_bonus = 9.0
            elif avg_response > 0.8 and consistency > 0.75:
                quality_bonus = 6.0
            elif avg_response > 0.75:
                quality_bonus = 3.0
                
            if recent_improvement > 0.6:
                quality_bonus *= 1.5
            
            base_reward += quality_bonus
            base_reward *= (1.0 + consistency * 0.5 + recent_improvement * 0.3 + overall_improvement * 0.2)
        
        done = self.current_step >= self.session_length
        
        return self._get_state(), base_reward, done

    def _get_state(self):
        state = self.track_history + self.response_history
        return state

class DQNAgent:
    """DQN agent for learning playlist generation policy"""
    def __init__(self, state_dim, action_feature_dim):
        self.policy_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=30,
            min_lr=5e-5,
            verbose=False
        )
        
        self.memory = ReplayBuffer()
        self.steps_done = 0
        self.max_grad_norm = 1.0

    def select_action(self, state, action_features, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                q_values = self.policy_net(state_tensor, action_features)
                
                state_array = np.array(state)
                track_history = state_array[:len(state_array)//2]
                action_penalties = torch.zeros_like(q_values)
                
                for i in range(action_features.size(0)):
                    if i in track_history:
                        action_penalties[0, i] = -1.0
                
                q_values += action_penalties
                return torch.argmax(q_values).item()
        else:
            return random.randrange(action_features.size(0))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        actual_batch_size = BATCH_SIZE
        
        states, actions, action_features, rewards, next_states, dones = self.memory.sample(actual_batch_size)
        
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        action_features = action_features.to(DEVICE)
        rewards = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)
        
        current_q = self.policy_net(states, action_features).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states, action_features).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

def save_models(user_simulator, agent, save_path='saved_models'):
    """モデルの保存"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(user_simulator.state_dict(), f'{save_path}/user_simulator.pth')
    torch.save(agent.policy_net.state_dict(), f'{save_path}/dqn_model.pth')

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard ログディレクトリ: {log_dir}")
    print(f"起動コマンド: tensorboard --logdir=runs")
    
    state_dim = 40
    action_feature_dim = 64
    track_pool_size = 1000
    session_length = state_dim // 2
    
    # LSTMユーザーシミュレータを初期化
    user_simulator = LSTMUserSimulator(state_dim + 1).to(DEVICE)
    
    # 事前学習を実行
    pretrain_user_simulator(user_simulator, n_episodes=500)
    
    # 事前学習済みモデルを保存
    os.makedirs('saved_models', exist_ok=True)
    torch.save(user_simulator.state_dict(), 'saved_models/user_simulator_pretrained.pth')
    print("事前学習済みシミュレータを保存しました\n")
    
    env = MusicEnvironment(user_simulator, track_pool_size, session_length, state_dim)
    agent = DQNAgent(state_dim, action_feature_dim)
    
    n_episodes = 300
    epsilon_start = 1.0
    epsilon_end = 0.10
    epsilon_decay = 0.996
    
    patience = 50
    min_delta = 1.0
    min_episodes = 100
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    cached_action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    rewards_history = []
    moving_avg_window = 10
    
    print("=" * 60)
    print("DQNエージェントの学習を開始...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        episode_step = 0
        
        while True:
            action_features = cached_action_features
            action = agent.select_action(state, action_features, epsilon)
            next_state, reward, done = env.step(action)
            
            selected_action_feat = action_features[action].cpu().detach()
            agent.memory.push(state, action, selected_action_feat, reward, next_state, done)
            
            loss = agent.train_step()
            total_loss += loss
            total_reward += reward
            
            global_step = episode * session_length + episode_step
            writer.add_scalar('Step/Reward', reward, global_step)
            writer.add_scalar('Step/Loss', loss, global_step)
            
            state = next_state
            episode_step += 1
            
            if done:
                rewards_history.append(total_reward)
                avg_reward = np.mean(rewards_history[-moving_avg_window:])
                
                agent.scheduler.step(avg_reward)
                
                writer.add_scalar('Episode/Total_Reward', total_reward, episode)
                writer.add_scalar('Episode/Average_Reward', avg_reward, episode)
                writer.add_scalar('Episode/Average_Loss', total_loss/session_length, episode)
                writer.add_scalar('Episode/Epsilon', epsilon, episode)
                writer.add_scalar('Episode/Learning_Rate', 
                                agent.optimizer.param_groups[0]['lr'], episode)
                
                avg_response = np.mean(env.response_history)
                response_std = np.std(env.response_history)
                
                writer.add_scalar('Environment/Average_Response', avg_response, episode)
                writer.add_scalar('Environment/Response_Std', response_std, episode)
                
                if episode >= min_episodes:
                    if avg_reward > best_avg_reward + min_delta:
                        best_avg_reward = avg_reward
                        patience_counter = 0
                        save_models(user_simulator, agent, save_path='saved_models_best')
                        print(f"✓ 新ベストモデル保存: {best_avg_reward:.2f}")
                    else:
                        patience_counter += 1
                elif avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                
                print(f"Ep {episode + 1:3d}, "
                      f"Total: {total_reward:6.2f}, "
                      f"Avg: {avg_reward:6.2f}, "
                      f"Loss: {total_loss/session_length:5.3f}, "
                      f"ε: {epsilon:.3f}, "
                      f"Resp: {avg_response:.3f}")

                if (episode + 1) % 50 == 0:
                    save_models(user_simulator, agent)

                if patience_counter >= patience:
                    print(f"\n早期停止 (最良: {best_avg_reward:.4f})")
                    save_models(user_simulator, agent, save_path='saved_models_final')
                    writer.close()
                    return

                break
    
    writer.close()
    print(f"\n学習完了! Logs: {log_dir}")

if __name__ == "__main__":
    main()