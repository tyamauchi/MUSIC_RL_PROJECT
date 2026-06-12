import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from config.config import DEVICE, PPO_LR, PPO_GAMMA, PPO_LAMBDA, PPO_EPS_CLIP, PPO_K_EPOCHS, PPO_VALUE_COEF, PPO_ENTROPY_COEF
from models.ppo import ActorNetwork, CriticNetwork


class PPOAgent:
    """PPO (Proximal Policy Optimization) エージェント"""
    
    def __init__(self, state_dim, action_feature_dim, use_ppo=True):
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.action_dim = 256  # TRACK_POOL_SIZE
        
        # Actor-Criticネットワーク
        self.actor = ActorNetwork(state_dim, action_feature_dim, self.action_dim).to(DEVICE)
        self.critic = CriticNetwork(state_dim).to(DEVICE)
        
        # オプティマイザー
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LR)
        
        # スケジューラー
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)
        
        # メモリ（エピソード単位）
        self.memory = {
            'states': [],
            'actions': [],
            'action_features': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        self.steps_done = 0
        self.episodes_done = 0
        
    def select_action(self, state, action_features, epsilon=None):
        """アクション選択（epsilonはDQN互換用、PPOでは無視）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # 履歴に基づくペナルティ
            track_history = np.array(state)[:20]
            
            action, log_prob = self.actor.get_action(state_tensor, action_features, track_history)
            
        return action
    
    def select_action_with_logprob(self, state, action_features):
        """アクション選択とlog_probを同時に返す（学習時用）"""
        # 収集時はeval mode（Dropout無効で確定的な分布）
        self.actor.eval()
        self.critic.eval()
        
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        track_history = np.array(state)[:20]
        
        # 確率分布を計算
        action_probs = self.actor(state_tensor, action_features)
        
        # 履歴にペナルティ
        penalties = torch.zeros_like(action_probs)
        for track_id in track_history:
            if 0 <= track_id < self.action_dim and track_id != 0:
                penalties[0, int(track_id)] = -0.5
        action_probs = action_probs + penalties
        action_probs = torch.clamp(action_probs, min=0.01)  # 最小確率を確保
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # 再正規化
        
        # サンプリング
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 元のtrain modeに戻す
        self.actor.train()
        self.critic.train()
        
        return action.item(), log_prob.detach().item()
    
    def store_transition(self, state, action, action_feature, reward, next_state, done, log_prob=None):
        """遷移をメモリに保存"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['action_features'].append(action_feature)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)
        if log_prob is not None:
            self.memory['log_probs'].append(log_prob)
        
        self.steps_done += 1
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=PPO_GAMMA, lambda_=PPO_LAMBDA):
        """Generalized Advantage Estimation (GAE) を計算"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        return advantages
    
    def train_step(self, action_features=None):
        """PPOの学習（エピソード終了時に呼び出し）
        
        Args:
            action_features: 全アクションの特徴量 [action_dim, action_feature_dim]
        """
        # 更新時はtrain mode（Dropout有効）
        self.actor.train()
        self.critic.train()
        
        if len(self.memory['states']) == 0:
            return 0.0, 0.0, 0.0
        
        # action_featuresが指定されていない場合はメモリから取得（後方互換）
        # 注意: メモリには全アクションプールの特徴量が保存されている前提
        if action_features is None:
            if len(self.memory['action_features']) > 0:
                # 全アクションプールの特徴量を取得（最初のステップに保存されている）
                action_features = torch.FloatTensor(self.memory['action_features'][0]).to(DEVICE)
            else:
                return 0.0, 0.0, 0.0
        
        # データをテンソルに変換
        states = torch.FloatTensor(np.array(self.memory['states'])).to(DEVICE)
        actions = torch.LongTensor(self.memory['actions']).to(DEVICE)
        rewards = torch.FloatTensor(self.memory['rewards']).to(DEVICE)
        next_states = torch.FloatTensor(np.array(self.memory['next_states'])).to(DEVICE)
        dones = torch.FloatTensor(self.memory['dones']).to(DEVICE)
        
        # action_featuresをテンソルに変換
        if not isinstance(action_features, torch.Tensor):
            action_features = torch.FloatTensor(action_features).to(DEVICE)
        
        # 現在の価値を計算
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
        
        # GAE計算
        advantages = self.compute_gae(
            self.memory['rewards'],
            values.cpu().numpy(),
            next_values.cpu().numpy(),
            self.memory['dones']
        )
        
        # リターン計算（価値ベースライン用）
        returns = advantages + values
        
        # アドバンテージ正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 古いlog_probを再計算（メモリに保存されている値を使用）
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(DEVICE)
        
        # PPO更新（複数エポック）
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(PPO_K_EPOCHS):
            # バッチ処理
            for i in range(len(self.memory['states'])):
                state_i = states[i:i+1]
                action_i = actions[i:i+1]
                advantage_i = advantages[i:i+1]
                return_i = returns[i:i+1]
                old_log_prob_i = old_log_probs[i:i+1]
                
                # 現在の方策でlog_probとエントロピーを計算（全アクション特徴量を使用）
                action_probs = self.actor(state_i, action_features)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(action_i)
                entropy = dist.entropy()
                
                # 比率計算
                ratio = torch.exp(log_prob - old_log_prob_i)
                
                # Surrogate loss（クリッピング）
                surr1 = ratio * advantage_i
                surr2 = torch.clamp(ratio, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * advantage_i
                actor_loss = -torch.min(surr1, surr2).mean() - PPO_ENTROPY_COEF * entropy.mean()
                
                # Actor更新
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic loss（価値関数）
                current_value = self.critic(state_i).squeeze(-1)  # [1, 1] -> [1]
                critic_loss = nn.MSELoss()(current_value, return_i)
                
                # Critic更新
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # n_updatesを計算（メモリクリア前に）
        n_updates = PPO_K_EPOCHS * len(self.memory['states'])
        
        # メモリクリア
        self.clear_memory()
        
        # スケジューラー更新
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 0除算防止
        if n_updates == 0:
            return 0.0, 0.0, 0.0
        
        return total_actor_loss / n_updates, total_critic_loss / n_updates, total_entropy / n_updates
    
    def clear_memory(self):
        """メモリをクリア"""
        for key in self.memory:
            self.memory[key] = []
    
    def update_target_network(self):
        """PPOではターゲットネットワーク更新不要だが、DQN互換用"""
        self.episodes_done += 1
        # PPOはオンライン更新なのでターゲットネットワーク不要
        pass
    
    def save_model(self, save_dir, metrics, action_features):
        """モデルを保存"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor_net.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic_net.pth'))
        torch.save(action_features, os.path.join(save_dir, 'action_features.pth'))
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_model(self, model_path):
        """モデルを読み込み
        
        Args:
            model_path: モデルディレクトリのパス（actor_net.pthとcritic_net.pthを含む）
        """
        actor_path = os.path.join(model_path, 'actor_net.pth')
        critic_path = os.path.join(model_path, 'critic_net.pth')
        self.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
