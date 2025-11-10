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
GAMMA = 0.995
LEARNING_RATE = 0.0002
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQ = 5
LSTM_HIDDEN_SIZE = [256, 128, 64]  # より小さいネットワーク

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
    """
    決定論的シミュレータで高品質な軌跡を生成
    
    重要: ランダムではなく、貪欲な選択をシミュレート
    """
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
            
            # 貪欲選択: 複数候補から最良の曲を選ぶ
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

class ActionHeadDQN(nn.Module):
    """Action Head DQN"""
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
        
        base_reward = 10 * np.log1p(response * 2)
        
        if self.current_step > 1:
            prev_response = self.response_history[self.current_step-2]
            curr_bonus = 0.0
            
            if prev_response > 0.9 and response > 0.9:
                curr_bonus = 2.0
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
            
            quality_bonus = 0.0
            if avg_response > 0.9:
                quality_bonus = 12.0
            elif avg_response > 0.85:
                quality_bonus = 9.0
            elif avg_response > 0.8:
                quality_bonus = 6.0
            
            base_reward += quality_bonus
            base_reward *= (1.0 + consistency * 0.5)
        
        done = self.current_step >= self.session_length
        return self._get_state(), base_reward, done

    def _get_state(self):
        return self.track_history + self.response_history

class DQNAgent:
    def __init__(self, state_dim, action_feature_dim):
        self.policy_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=30, min_lr=5e-5
        )
        
        self.memory = ReplayBuffer()
        self.steps_done = 0

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
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        states, actions, action_features, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = states.to(DEVICE)
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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    print("\n" + "=" * 60)
    print("ハイブリッドアプローチによる学習")
    print("=" * 60)
    
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
    
    # Step 3: DQN学習
    env = MusicEnvironment(lstm_sim, track_pool_size, session_length, state_dim)
    agent = DQNAgent(state_dim, action_feature_dim)
    
    n_episodes = 300
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.995
    best_avg_reward = float('-inf')
    rewards_history = []
    cached_action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    print("\n" + "=" * 60)
    print("DQN学習開始")
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
        
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-10:])
        agent.scheduler.step(avg_reward)
        
        avg_resp = np.mean(env.response_history)
        
        writer.add_scalar('Episode/Total_Reward', total_reward, episode)
        writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Episode/Avg_Response', avg_resp, episode)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:3d} | Total: {total_reward:6.1f} | "
                  f"Avg: {avg_reward:6.1f} | Resp: {avg_resp:.3f} | ε: {epsilon:.3f}")
    
    writer.close()
    print(f"\n学習完了! 最良平均報酬: {best_avg_reward:.2f}")

if __name__ == "__main__":
    main()