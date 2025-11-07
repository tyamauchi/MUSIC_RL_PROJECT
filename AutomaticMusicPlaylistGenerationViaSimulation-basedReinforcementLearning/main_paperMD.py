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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)

class ActionHeadDQN(nn.Module):
    """Action Head DQN for playlist generation"""
    def __init__(self, state_dim, action_feature_dim):
        super(ActionHeadDQN, self).__init__()
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
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
        # ユーザーの気分状態（エピソードごとに変化）
        self.user_mood = np.random.uniform(0.6, 1.0)
        self.user_pickiness = np.random.uniform(0.0, 0.3)  # 気難しさ
        
    def reset(self):
        self.current_step = 0
        self.track_history = [0.0] * (self.state_dim // 2)
        self.response_history = [0.0] * (self.state_dim // 2)
        # エピソードごとに異なるユーザー特性を設定
        self.user_mood = np.random.uniform(0.6, 1.0)
        self.user_pickiness = np.random.uniform(0.0, 0.3)
        return self._get_state()

    def step(self, action):
        state = self._get_state()
        state_tensor = torch.FloatTensor([state + [float(action)]]).unsqueeze(0)
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
        # 固定学習率スケジューラーに変更（ReduceLROnPlateauを使わない）
        # Cosine Annealing with Warm Restartsを使用
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,      # 50エピソードごとに学習率をリセット
            T_mult=1,    # リスタート間隔を一定に保つ
            eta_min=5e-5 # 最小学習率
        )
        self.memory = ReplayBuffer()
        self.steps_done = 0
        self.max_grad_norm = 1.0

    def select_action(self, state, action_features, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.policy_net(state_tensor, action_features)
                
                state_array = np.array(state)
                track_history = state_array[:len(state_array)//2]
                action_penalties = torch.zeros_like(q_values)
                
                for i in range(action_features.size(0)):
                    if i in track_history:
                        action_penalties[0, i] = -0.5
                
                q_values += action_penalties
                return torch.argmax(q_values).item()
        else:
            return random.randrange(action_features.size(0))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        if len(self.memory) < INITIAL_MEMORY:
            actual_batch_size = min(48, len(self.memory))
        else:
            actual_batch_size = min(BATCH_SIZE, len(self.memory))
            
        if self.steps_done % 50 == 0:
            actual_batch_size = min(128, len(self.memory))
            
        if len(self.memory) > actual_batch_size * 2:
            actual_batch_size = int(actual_batch_size * 1.5)
        
        total_loss = 0.0
        n_updates = 1
        
        for _ in range(n_updates):
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
    # TensorBoard writerの初期化
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard ログディレクトリ: {log_dir}")
    print(f"起動コマンド: tensorboard --logdir=runs")
    
    # Initialize parameters
    state_dim = 40
    action_feature_dim = 64
    track_pool_size = 1000
    session_length = state_dim // 2
    
    # User Simulatorを初期化し、より現実的な応答を生成するように設定
    user_simulator = LSTMUserSimulator(state_dim + 1).to(DEVICE)
    
    # User Simulatorに事前学習的な初期化を適用（バイアスを調整して高い応答率を促進）
    with torch.no_grad():
        # 出力層の最終バイアスを調整して、初期応答率を高める
        user_simulator.output_layer[-2].bias.fill_(2.0)  # Sigmoid前のバイアスを正の値に
    
    env = MusicEnvironment(user_simulator, track_pool_size, session_length, state_dim)
    agent = DQNAgent(state_dim, action_feature_dim)
    
    # Training parameters
    n_episodes = 300
    epsilon_start = 1.0
    epsilon_end = 0.35  # さらに高い最小探索率
    epsilon_decay = 0.998  # さらに緩やかな減衰
    
    patience = 50
    min_delta = 0.0005
    min_episodes = 100
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    cached_action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    rewards_history = []
    moving_avg_window = 10
    
    # TensorBoard用の追加メトリクス
    step_rewards = []
    step_losses = []
    
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
            
            try:
                selected_action_feat = action_features[action].cpu().detach()
            except Exception:
                selected_action_feat = action_features.cpu().detach()
            agent.memory.push(state, action, selected_action_feat, reward, next_state, done)
            
            loss = agent.train_step()
            total_loss += loss
            total_reward += reward
            
            # ステップごとのメトリクスを記録
            step_rewards.append(reward)
            step_losses.append(loss)
            
            # TensorBoard: ステップごとのメトリクス
            global_step = episode * session_length + episode_step
            writer.add_scalar('Step/Reward', reward, global_step)
            writer.add_scalar('Step/Loss', loss, global_step)
            writer.add_scalar('Step/Q_Value_Mean', 
                            agent.policy_net(torch.FloatTensor(state).to(DEVICE), 
                                           action_features).mean().item(), 
                            global_step)
            
            state = next_state
            episode_step += 1
            
            if done:
                rewards_history.append(total_reward)
                avg_reward = np.mean(rewards_history[-moving_avg_window:])
                
                agent.scheduler.step(avg_reward)
                
                # TensorBoard: エピソードごとのメトリクス
                writer.add_scalar('Episode/Total_Reward', total_reward, episode)
                writer.add_scalar('Episode/Average_Reward', avg_reward, episode)
                writer.add_scalar('Episode/Average_Loss', total_loss/session_length, episode)
                writer.add_scalar('Episode/Epsilon', epsilon, episode)
                writer.add_scalar('Episode/Learning_Rate', 
                                agent.optimizer.param_groups[0]['lr'], episode)
                writer.add_scalar('Episode/Memory_Size', len(agent.memory), episode)
                
                # TensorBoard: 環境メトリクスを詳細に記録
                avg_response = np.mean(env.response_history)
                response_std = np.std(env.response_history)
                response_min = np.min(env.response_history)
                response_max = np.max(env.response_history)
                response_range = response_max - response_min
                
                writer.add_scalar('Environment/Average_Response', avg_response, episode)
                writer.add_scalar('Environment/Response_Std', response_std, episode)
                writer.add_scalar('Environment/Response_Min', response_min, episode)
                writer.add_scalar('Environment/Response_Max', response_max, episode)
                writer.add_scalar('Environment/Response_Range', response_range, episode)
                writer.add_scalar('Environment/Response_Consistency', 1.0 - response_std, episode)
                writer.add_scalar('Environment/User_Mood', env.user_mood, episode)
                writer.add_scalar('Environment/User_Pickiness', env.user_pickiness, episode)
                
                # ヒストグラム: 応答の分布
                writer.add_histogram('Distribution/Response_History', 
                                    np.array(env.response_history), episode)
                writer.add_histogram('Distribution/Step_Rewards', 
                                    np.array(step_rewards[-session_length:]), episode)
                
                # ネットワークパラメータのヒストグラム
                for name, param in agent.policy_net.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param, episode)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, episode)
                
                # 早期停止の判定
                if episode >= min_episodes:
                    if avg_reward > best_avg_reward + min_delta:
                        best_avg_reward = avg_reward
                        patience_counter = 0
                        save_models(user_simulator, agent, save_path='saved_models_best')
                        writer.add_scalar('Milestone/Best_Model_Update', best_avg_reward, episode)
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

                if (episode + 1) % 50 == 0:
                    save_models(user_simulator, agent)
                    writer.add_text('Checkpoint', f'Model saved at episode {episode + 1}', episode)

                if patience_counter >= patience:
                    print(f"\n早期停止: {patience}エピソード連続で改善が見られませんでした")
                    print(f"最良の平均報酬: {best_avg_reward:.4f}")
                    save_models(user_simulator, agent, save_path='saved_models_best')
                    writer.add_text('Training', 
                                  f'Early stopping at episode {episode + 1}. Best reward: {best_avg_reward:.4f}', 
                                  episode)
                    writer.close()
                    return

                break
    
    writer.close()
    print(f"\nTrening completed. TensorBoard logs saved to: {log_dir}")
    print(f"View results with: tensorboard --logdir=runs")

if __name__ == "__main__":
    main()