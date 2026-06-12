"""
Heterogeneous Graph DQN agent (HGT-based Q-network).
API mirrors SimpleGraphDQNAgent for fair comparison in main.py / evaluation.
"""

import json
import os
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import (
    BATCH_SIZE,
    DEVICE,
    GAMMA,
    GRAPH_HIDDEN_DIM,
    HETERO_HGT_HEADS,
    LEARNING_RATE,
    MEMORY_SIZE,
    UPDATE_TARGET_FREQ,
)
from models.hetero_graph_dqn import HeteroGraphDQN
from utils.hetero_graph_builder import MusicHeteroGraphBuilder


EdgeKey = Tuple[str, str, str]


class HeteroGraphDQNAgent:
    """RL agent using HeteroGraphDQN over (track, artist, genre) graph."""

    def __init__(self, state_dim: int, track_metadata_path: str, data_source: str = "lastfm"):
        self.graph_builder = MusicHeteroGraphBuilder(track_metadata_path)
        self.metadata, x_cpu, e_cpu = self.graph_builder.get_graph_tensors()
        self.data_source = data_source
        self.agent_type = "hetero_graph_dqn"
        self.num_nodes = self.graph_builder.num_tracks  # RL actions = track indices

        self.x_dict: Dict[str, torch.Tensor] = {k: v.to(DEVICE) for k, v in x_cpu.items()}
        self.edge_index_dict: Dict[EdgeKey, torch.Tensor] = {
            k: v.to(DEVICE) for k, v in e_cpu.items()
        }

        in_channels_dict = {k: int(v.size(1)) for k, v in self.x_dict.items()}

        self.policy_net = HeteroGraphDQN(
            state_dim=state_dim,
            metadata=self.metadata,
            in_channels_dict=in_channels_dict,
            hidden_dim=GRAPH_HIDDEN_DIM,
            heads=HETERO_HGT_HEADS,
        ).to(DEVICE)

        self.target_net = HeteroGraphDQN(
            state_dim=state_dim,
            metadata=self.metadata,
            in_channels_dict=in_channels_dict,
            hidden_dim=GRAPH_HIDDEN_DIM,
            heads=HETERO_HGT_HEADS,
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.7, patience=30, min_lr=5e-5
        )

        self.memory: deque = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.episodes_done = 0

        print("🧩 Hetero Graph DQN エージェント初期化完了")
        print(f"  データソース: {data_source}")
        print(f"  トラックノード数 (行動空間グラフ): {self.num_nodes}")
        print(f"  HGT heads: {HETERO_HGT_HEADS}, hidden_dim: {GRAPH_HIDDEN_DIM}")

    def select_action(self, state, candidate_track_ids, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                candidate_nodes = [int(tid) for tid in candidate_track_ids]
                q_values = self.policy_net(
                    state_tensor,
                    self.x_dict,
                    self.edge_index_dict,
                    candidate_nodes,
                )
                track_history = np.array(state)[:20]
                penalties = torch.zeros_like(q_values)
                for i, track_id in enumerate(candidate_track_ids):
                    if track_id in track_history:
                        penalties[0, i] = -2.0
                adjusted_q_values = q_values + penalties
                return torch.argmax(adjusted_q_values).item()
        return random.randrange(len(candidate_track_ids))

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        current_q = self.policy_net(
            states,
            self.x_dict,
            self.edge_index_dict,
            actions.tolist(),
        ).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(
                next_states,
                self.x_dict,
                self.edge_index_dict,
                list(range(self.num_nodes)),
            )
            next_actions = next_q_policy.argmax(dim=1)
            next_q_target = self.target_net(
                next_states,
                self.x_dict,
                self.edge_index_dict,
                next_actions.tolist(),
            ).squeeze(1)
            target_q = rewards + GAMMA * next_q_target * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps_done += 1
        return loss.item()

    def update_target_network(self):
        self.episodes_done += 1
        if self.episodes_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  🔄 (Hetero) ターゲットネットワーク更新 (Episode {self.episodes_done})")

    def get_recommendations(self, state, candidate_track_ids, top_k=5):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            candidate_nodes = [int(tid) for tid in candidate_track_ids]
            q_values = self.policy_net(
                state_tensor,
                self.x_dict,
                self.edge_index_dict,
                candidate_nodes,
            )
            top_indices = torch.topk(q_values, min(top_k, len(candidate_track_ids))).indices
            out = []
            for idx in top_indices:
                tid = candidate_track_ids[idx.item()]
                out.append((tid, q_values[0, idx.item()].item()))
            return out

    def save_model(self, save_dir, metrics):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(save_dir, "policy_net.pth"))
        torch.save(self.target_net.state_dict(), os.path.join(save_dir, "target_net.pth"))
        bundle = {
            "metadata": self.metadata,
            "x_dict": {k: v.cpu() for k, v in self.x_dict.items()},
            "edge_index_dict": {k: v.cpu() for k, v in self.edge_index_dict.items()},
        }
        torch.save(bundle, os.path.join(save_dir, "hetero_graph_bundle.pth"))
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Hetero モデル保存完了: {save_dir}")

    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"📂 Hetero モデル読み込み: {model_path}")

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        n_edges = sum(e.shape[1] for e in self.edge_index_dict.values())
        return {
            "model_type": "HeteroGraphDQN",
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "num_tracks": self.num_nodes,
            "num_edges_directed": n_edges,
            "data_source": self.data_source,
            "agent_type": self.agent_type,
        }
