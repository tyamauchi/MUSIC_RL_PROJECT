import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """PPO Actor Network - 方策ネットワーク（離散アクション用）"""
    
    def __init__(self, state_dim, action_feature_dim, action_dim=256):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        
        # 状態エンコーダー
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # アクションフィーチャーエンコーダー
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 結合してアクション確率を出力
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state, action_features):
        """
        Args:
            state: [batch_size, state_dim]
            action_features: [action_dim, action_feature_dim] または [batch_size, action_dim, action_feature_dim]
        Returns:
            action_probs: [batch_size, action_dim] - 各アクションの確率分布
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 状態エンコード
        state_features = self.state_net(state)  # [batch_size, 128]
        
        # アクションフィーチャーの整形
        if action_features.dim() == 2:
            # [action_dim, action_feature_dim] -> [1, action_dim, action_feature_dim]
            action_features = action_features.unsqueeze(0)
        
        batch_size = state_features.size(0)
        num_actions = action_features.size(1)
        
        # アクションフィーチャーをエンコード
        action_features_flat = action_features.view(-1, action_features.size(-1))  # [batch_size*action_dim, action_feature_dim]
        action_features_encoded = self.action_net(action_features_flat)  # [batch_size*action_dim, 64]
        action_features_encoded = action_features_encoded.view(batch_size, num_actions, -1)  # [batch_size, action_dim, 64]
        
        # 状態を各アクションに展開
        state_features_expanded = state_features.unsqueeze(1).expand(-1, num_actions, -1)  # [batch_size, action_dim, 128]
        
        # 結合
        combined = torch.cat([state_features_expanded, action_features_encoded], dim=-1)  # [batch_size, action_dim, 192]
        combined_flat = combined.view(batch_size * num_actions, -1)  # [batch_size*action_dim, 192]
        
        # スコア計算
        scores = self.policy_head(combined_flat).view(batch_size, num_actions)  # [batch_size, action_dim]
        
        # ソフトマックスで確率分布に
        action_probs = F.softmax(scores, dim=-1)
        
        return action_probs
    
    def get_action(self, state, action_features, track_history=None):
        """アクションをサンプリングして返す"""
        with torch.no_grad():
            action_probs = self.forward(state, action_features)
            
            # 履歴にあるトラックにペナルティ（select_action_with_logprobと同じ処理）
            if track_history is not None:
                penalties = torch.zeros_like(action_probs)
                for track_id in track_history:
                    track_id_int = int(track_id)
                    if 0 <= track_id_int < self.action_dim and track_id_int != 0:
                        penalties[0, track_id_int] = -0.5
                action_probs = action_probs + penalties
                action_probs = torch.clamp(action_probs, min=0.01)  # 最小確率を確保
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # 再正規化
            
            # 確率分布からサンプリング
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()


class CriticNetwork(nn.Module):
    """PPO Critic Network - 価値ネットワーク"""
    
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] または [state_dim]
        Returns:
            value: [batch_size, 1] または [1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        return self.value_net(state)
