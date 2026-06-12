"""
Simple Graph DQNモデル
DuelingActionHeadDQNのaction_net部分をGraph Convolutionに置き換え
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGraphDQN(nn.Module):
    """
    Simple Graph DQN
    グラフ畳み込みを用いたQ関数近似
    """
    
    def __init__(self, state_dim, num_nodes, node_feature_dim, hidden_dim=128):
        """
        初期化
        
        Args:
            state_dim: 状態次元
            num_nodes: ノード数
            node_feature_dim: ノード特徴量次元
            hidden_dim: 隠れ層次元
        """
        super(SimpleGraphDQN, self).__init__()
        
        # 状態処理ネットワーク（既存と同じ）
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # グラフ畳み込み層（action_netの置き換え）
        self.graph_conv = GCNConv(node_feature_dim, hidden_dim)
        self.graph_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Duelingアーキテクチャ（既存と同じ）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim + 128, 64),  # グラフ埋め込み + 状態特徴
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim + 128, 64),  # グラフ埋め込み + 状態特徴
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, node_features, edge_index, candidate_nodes):
        """
        Forward pass
        
        Args:
            state: 状態 [batch_size, state_dim]
            node_features: ノード特徴量 [num_nodes, node_feature_dim]
            edge_index: エッジインデックス [2, num_edges]
            candidate_nodes: 候補ノードインデックスリスト [num_candidates]
        
        Returns:
            q_values: Q値 [batch_size, num_candidates]
        """
        # 状態処理
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state_features = self.state_net(state)  # [batch_size, 128]
        batch_size = state_features.size(0)
        
        # グラフ畳み込み
        x = F.relu(self.graph_conv(node_features, edge_index))
        x = F.relu(self.graph_conv2(x, edge_index))  # [num_nodes, hidden_dim]
        
        # 候補ノードの埋め込みを取得
        candidate_emb = x[candidate_nodes]  # [num_candidates, hidden_dim]
        
        # 状態特徴と候補埋め込みを組み合わせ
        candidate_nodes_count = len(candidate_nodes)
        
        # 状態特徴を候補数分に拡張
        state_expanded = state_features.unsqueeze(1).expand(
            batch_size, candidate_nodes_count, -1
        )  # [batch_size, num_candidates, 128]
        
        # 候補埋め込みをバッチ次元に拡張
        candidate_emb_expanded = candidate_emb.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_candidates, hidden_dim]
        
        # 状態とグラフ埋め込みを結合
        combined = torch.cat([
            state_expanded, 
            candidate_emb_expanded
        ], dim=-1)  # [batch_size, num_candidates, hidden_dim + 128]
        
        # Dueling DQN計算
        value = self.value_stream(combined)  # [batch_size, num_candidates, 1]
        advantage = self.advantage_stream(combined)  # [batch_size, num_candidates, 1]
        
        # Q値計算
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values.squeeze(-1)  # [batch_size, num_candidates]
    
    def get_node_embeddings(self, node_features, edge_index):
        """
        ノード埋め込みを取得
        
        Args:
            node_features: ノード特徴量
            edge_index: エッジインデックス
        
        Returns:
            embeddings: ノード埋め込み
        """
        x = F.relu(self.graph_conv(node_features, edge_index))
        x = F.relu(self.graph_conv2(x, edge_index))
        return x


if __name__ == "__main__":
    # テスト用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル初期化
    model = SimpleGraphDQN(
        state_dim=40,
        num_nodes=256,
        node_feature_dim=3,
        hidden_dim=128
    ).to(device)
    
    # ダミーデータ
    state = torch.randn(1, 40).to(device)
    node_features = torch.randn(256, 3).to(device)
    edge_index = torch.randint(0, 256, (2, 1000)).to(device)
    candidate_nodes = [0, 1, 2, 3, 4]
    
    # Forward pass
    with torch.no_grad():
        q_values = model(state, node_features, edge_index, candidate_nodes)
        print(f"Q値形状: {q_values.shape}")
        print(f"Q値: {q_values}")
