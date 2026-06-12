"""
Heterogeneous Graph DQN — Q-network with HGTConv over (track, artist, genre).

Compared to SimpleGraphDQN (homogeneous GCN on track–track edges), this model
uses explicit artist/genre nodes and typed edges from track_metadata.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv

Metadata = Tuple[List[str], List[Tuple[str, str, str]]]


class HeteroGraphDQN(nn.Module):
    """
    State encoder + two-layer HGT over heterogeneous music graph.
    Q(s, a) for candidate track nodes only (same interface style as SimpleGraphDQN).
    """

    def __init__(
        self,
        state_dim: int,
        metadata: Metadata,
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.conv1 = HGTConv(
            in_channels_dict,
            hidden_dim,
            metadata,
            heads=heads,
        )
        h_in = {nt: hidden_dim for nt in self.node_types}
        self.conv2 = HGTConv(
            h_in,
            hidden_dim,
            metadata,
            heads=heads,
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _encode_graph(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Returns track node embeddings [num_tracks, hidden_dim]."""
        h = self.conv1(x_dict, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        return h["track"]

    def forward(
        self,
        state: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        candidate_nodes: List[int],
    ) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] or [state_dim]
            x_dict: node features per type (on same device as self)
            edge_index_dict: PyG heterogeneous edge indices
            candidate_nodes: local track indices into graph (0 .. num_tracks-1)

        Returns:
            q_values: [batch_size, len(candidate_nodes)]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        state_features = self.state_net(state)
        batch_size = state_features.size(0)

        x_track = self._encode_graph(x_dict, edge_index_dict)
        candidate_emb = x_track[candidate_nodes]
        candidate_nodes_count = len(candidate_nodes)

        state_expanded = state_features.unsqueeze(1).expand(
            batch_size, candidate_nodes_count, -1
        )
        candidate_emb_expanded = candidate_emb.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        combined = torch.cat([state_expanded, candidate_emb_expanded], dim=-1)
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values.squeeze(-1)

    def get_track_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Track embeddings after HGT (for analysis / optional use)."""
        with torch.no_grad():
            return self._encode_graph(x_dict, edge_index_dict)


if __name__ == "__main__":
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from utils.hetero_graph_builder import build_hetero_music_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(root, "data/hetrec2011-lastfm/track_metadata.json")
    g = build_hetero_music_graph(meta_path)
    in_dim = {k: v.size(1) for k, v in g.x_dict.items()}
    model = HeteroGraphDQN(
        state_dim=40,
        metadata=g.metadata,
        in_channels_dict=in_dim,
        hidden_dim=64,
        heads=2,
    ).to(device)

    xd = {k: v.to(device) for k, v in g.x_dict.items()}
    ed = {k: v.to(device) for k, v in g.edge_index_dict.items()}
    st = torch.randn(2, 40, device=device)
    cand = [0, 1, 2, 5]
    q = model(st, xd, ed, cand)
    print("Q shape", q.shape, "expected", (2, len(cand)))
