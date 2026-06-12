"""
Simple Graph DQNエージェント
グラフ畳み込みを用いた強化学習エージェント
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from models.simple_graph_dqn import SimpleGraphDQN
from utils.graph_builder import MusicGraphBuilder
from config.config import *


class SimpleGraphDQNAgent:
    """
    Simple Graph DQNエージェント
    DuelingActionHeadDQNのaction_netをGraph Convolutionに置き換え
    """
    
    def __init__(self, state_dim, track_metadata_path, data_source="lastfm"):
        """
        初期化
        
        Args:
            state_dim: 状態次元
            track_metadata_path: track_metadata.jsonのパス
            data_source: データソース ("lastfm" or "det_sim")
        """
        # グラフ構築
        self.graph_builder = MusicGraphBuilder(track_metadata_path)
        self.edge_index, self.node_features = self.graph_builder.get_graph_data()
        
        # データソース情報
        self.data_source = data_source
        self.num_nodes = self.graph_builder.num_nodes
        
        # モデル初期化
        self.policy_net = SimpleGraphDQN(
            state_dim=state_dim,
            num_nodes=self.num_nodes,
            node_feature_dim=3,  # popularity, duration, year
            hidden_dim=GRAPH_HIDDEN_DIM
        ).to(DEVICE)
        
        self.target_net = SimpleGraphDQN(
            state_dim=state_dim,
            num_nodes=self.num_nodes,
            node_feature_dim=3,
            hidden_dim=GRAPH_HIDDEN_DIM
        ).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # オプティマイザとスケジューラ
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=30, min_lr=5e-5
        )
        
        # 学習パラメータ
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.episodes_done = 0
        
        print(f"🚀 Simple Graph DQNエージェント初期化完了")
        print(f"  データソース: {data_source}")
        print(f"  ノード数: {self.num_nodes}")
        print(f"  状態次元: {state_dim}")
    
    def select_action(self, state, candidate_track_ids, epsilon):
        """
        行動選択（ε-greedy）
        
        Args:
            state: 状態
            candidate_track_ids: 候補トラックIDリスト
            epsilon: 探索率
        
        Returns:
            action: 選択された行動インデックス
        """
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                
                # 候補トラックのノードインデックス
                candidate_nodes = [int(tid) for tid in candidate_track_ids]
                
                # Q値計算
                q_values = self.policy_net(
                    state_tensor,
                    self.node_features.to(DEVICE),
                    self.edge_index.to(DEVICE),
                    candidate_nodes
                )
                
                # 履歴ペナルティ（既存ロジックと同じ）
                track_history = np.array(state)[:20]
                penalties = torch.zeros_like(q_values)
                for i, track_id in enumerate(candidate_track_ids):
                    if track_id in track_history:
                        penalties[0, i] = -2.0
                
                # ペナルティを適用したQ値
                adjusted_q_values = q_values + penalties
                
                return torch.argmax(adjusted_q_values).item()
        else:
            return random.randrange(len(candidate_track_ids))
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        遷移を保存
        
        Args:
            state: 現在の状態
            action: 行動
            reward: 報酬
            next_state: 次の状態
            done: 終了フラグ
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        学習ステップ
        
        Returns:
            loss: 損失値
        """
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        # バッチサンプリング
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # 現在のQ値
        current_q = self.policy_net(
            states, 
            self.node_features.to(DEVICE),
            self.edge_index.to(DEVICE),
            actions.tolist()
        ).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            # ポリシーネットワークで次の行動を選択
            next_q_policy = self.policy_net(
                next_states,
                self.node_features.to(DEVICE),
                self.edge_index.to(DEVICE),
                list(range(self.num_nodes))  # 全ノードを候補として
            )
            next_actions = next_q_policy.argmax(dim=1)
            
            # ターゲットネットワークでQ値を評価
            next_q_target = self.target_net(
                next_states,
                self.node_features.to(DEVICE),
                self.edge_index.to(DEVICE),
                next_actions.tolist()
            ).squeeze(1)
            
            target_q = rewards + GAMMA * next_q_target * (1 - dones)
        
        # 損失計算
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # 最適化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        return loss.item()
    
    def update_target_network(self):
        """ターゲットネットワークの更新"""
        self.episodes_done += 1
        if self.episodes_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  🔄 ターゲットネットワーク更新 (Episode {self.episodes_done})")
    
    def get_recommendations(self, state, candidate_track_ids, top_k=5):
        """
        推薦を取得
        
        Args:
            state: 状態
            candidate_track_ids: 候補トラックIDリスト
            top_k: 上位k件
        
        Returns:
            recommendations: 推薦リスト [(track_id, score), ...]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            candidate_nodes = [int(tid) for tid in candidate_track_ids]
            
            q_values = self.policy_net(
                state_tensor,
                self.node_features.to(DEVICE),
                self.edge_index.to(DEVICE),
                candidate_nodes
            )
            
            # 上位k件を取得
            top_indices = torch.topk(q_values, min(top_k, len(candidate_track_ids))).indices
            
            recommendations = []
            for idx in top_indices:
                track_id = candidate_track_ids[idx.item()]
                score = q_values[0, idx.item()].item()
                recommendations.append((track_id, score))
            
            return recommendations
    
    def save_model(self, save_dir, metrics):
        """
        モデルを保存
        
        Args:
            save_dir: 保存ディレクトリ
            metrics: 評価指標
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # モデル保存
        torch.save(self.policy_net.state_dict(), os.path.join(save_dir, 'policy_net.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(save_dir, 'target_net.pth'))
        
        # グラフデータ保存
        torch.save(self.edge_index, os.path.join(save_dir, 'edge_index.pth'))
        torch.save(self.node_features, os.path.join(save_dir, 'node_features.pth'))
        
        # メトリクス保存
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 モデル保存完了: {save_dir}")
    
    def load_model(self, model_path):
        """
        モデルを読み込み
        
        Args:
            model_path: モデルパス
        """
        self.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"📂 モデル読み込み完了: {model_path}")
    
    def get_model_info(self):
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        trainable_params = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        
        return {
            'model_type': 'SimpleGraphDQN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_nodes': self.num_nodes,
            'num_edges': self.edge_index.shape[1],
            'data_source': self.data_source
        }


if __name__ == "__main__":
    # テスト用
    track_metadata_path = "./data/hetrec2011-lastfm/track_metadata.json"
    
    # エージェント初期化
    agent = SimpleGraphDQNAgent(
        state_dim=40,
        track_metadata_path=track_metadata_path,
        data_source="det_sim"
    )
    
    # モデル情報
    info = agent.get_model_info()
    print("モデル情報:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # テスト用の行動選択
    state = np.random.randn(40)
    candidate_tracks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    action = agent.select_action(state, candidate_tracks, epsilon=0.0)
    print(f"選択された行動: {action}")
    
    # 推薦テスト
    recommendations = agent.get_recommendations(state, candidate_tracks, top_k=3)
    print("推薦結果:")
    for track_id, score in recommendations:
        print(f"  トラック{track_id}: スコア{score:.4f}")
