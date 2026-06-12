"""
Heterogeneous music graph from track_metadata.json.

Node types:
  - track: one per metadata entry (RL action indices = track local id 0..N-1)
  - artist: unique artist strings from metadata
  - genre: unique genre strings from metadata

Edge types (bidirectional pairs for HGT message passing):
  - (track, belongs, artist) / (artist, rev_belongs, track)
  - (track, in_genre, genre) / (genre, rev_in_genre, track)
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


Metadata = Tuple[List[str], List[Tuple[str, str, str]]]


def _track_feature_vec(track_info: dict) -> List[float]:
    return [
        float(track_info.get("popularity", 0)) / 100.0,
        float(track_info.get("duration_ms", 0)) / 300000.0,
        float(track_info.get("release_year", 2000)) / 2020.0,
    ]


@dataclass
class HeteroMusicGraph:
    """Container for heterogeneous graph tensors (CPU by default)."""

    metadata: Metadata
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    num_tracks: int
    num_artists: int
    num_genres: int

    def to_device(self, device) -> "HeteroMusicGraph":
        return HeteroMusicGraph(
            metadata=self.metadata,
            x_dict={k: v.to(device) for k, v in self.x_dict.items()},
            edge_index_dict={k: v.to(device) for k, v in self.edge_index_dict.items()},
            num_tracks=self.num_tracks,
            num_artists=self.num_artists,
            num_genres=self.num_genres,
        )


class MusicHeteroGraphBuilder:
    """
    Build a heterogeneous graph from track_metadata.json (same file as homogeneous MusicGraphBuilder).
    """

    NODE_TYPES = ["track", "artist", "genre"]
    EDGE_TYPES: List[Tuple[str, str, str]] = [
        ("track", "belongs", "artist"),
        ("artist", "rev_belongs", "track"),
        ("track", "in_genre", "genre"),
        ("genre", "rev_in_genre", "track"),
    ]

    def __init__(self, track_metadata_path: str):
        with open(track_metadata_path, "r", encoding="utf-8") as f:
            self.track_data = json.load(f)

        # Stable integer order 0..N-1 for tracks
        sorted_ids = sorted(self.track_data.keys(), key=lambda x: int(x))
        self.num_tracks = len(sorted_ids)
        self._id_to_local = {int(k): i for i, k in enumerate(sorted_ids)}

        artist_to_idx: Dict[str, int] = {}
        genre_to_idx: Dict[str, int] = {}

        for k in sorted_ids:
            info = self.track_data[k]
            a = str(info.get("artist", "unknown"))
            g = str(info.get("genre", "unknown"))
            if a not in artist_to_idx:
                artist_to_idx[a] = len(artist_to_idx)
            if g not in genre_to_idx:
                genre_to_idx[g] = len(genre_to_idx)

        self.num_artists = len(artist_to_idx)
        self.num_genres = len(genre_to_idx)

        # --- node features (dim 3 for all types, comparable to homogeneous graph) ---
        x_track = torch.zeros(self.num_tracks, 3, dtype=torch.float32)
        for k in sorted_ids:
            tid = int(k)
            li = self._id_to_local[tid]
            x_track[li] = torch.tensor(_track_feature_vec(self.track_data[k]), dtype=torch.float32)

        # Artist / genre: mean-pool features from incident tracks
        artist_feats = [[] for _ in range(self.num_artists)]
        genre_feats = [[] for _ in range(self.num_genres)]
        for k in sorted_ids:
            info = self.track_data[k]
            li = self._id_to_local[int(k)]
            vec = x_track[li].tolist()
            ai = artist_to_idx[str(info.get("artist", "unknown"))]
            gi = genre_to_idx[str(info.get("genre", "unknown"))]
            artist_feats[ai].append(vec)
            genre_feats[gi].append(vec)

        def _mean_pool(rows: List[List[List[float]]]) -> torch.Tensor:
            out = torch.zeros(len(rows), 3, dtype=torch.float32)
            for i, vecs in enumerate(rows):
                if vecs:
                    out[i] = torch.tensor(vecs, dtype=torch.float32).mean(dim=0)
            return out

        x_artist = _mean_pool(artist_feats)
        x_genre = _mean_pool(genre_feats)

        self.x_dict = {
            "track": x_track,
            "artist": x_artist,
            "genre": x_genre,
        }

        # --- edges ---
        src_belongs: List[int] = []
        dst_belongs: List[int] = []
        src_genre: List[int] = []
        dst_genre: List[int] = []

        for k in sorted_ids:
            info = self.track_data[k]
            t_local = self._id_to_local[int(k)]
            ai = artist_to_idx[str(info.get("artist", "unknown"))]
            gi = genre_to_idx[str(info.get("genre", "unknown"))]
            src_belongs.append(t_local)
            dst_belongs.append(ai)
            src_genre.append(t_local)
            dst_genre.append(gi)

        e_belongs = torch.tensor([src_belongs, dst_belongs], dtype=torch.long)
        e_rev_belongs = torch.tensor([dst_belongs, src_belongs], dtype=torch.long)
        e_in_genre = torch.tensor([src_genre, dst_genre], dtype=torch.long)
        e_rev_in_genre = torch.tensor([dst_genre, src_genre], dtype=torch.long)

        self.edge_index_dict = {
            ("track", "belongs", "artist"): e_belongs,
            ("artist", "rev_belongs", "track"): e_rev_belongs,
            ("track", "in_genre", "genre"): e_in_genre,
            ("genre", "rev_in_genre", "track"): e_rev_in_genre,
        }

        self.metadata: Metadata = (list(self.NODE_TYPES), list(self.EDGE_TYPES))

        n_e = sum(e.shape[1] for e in self.edge_index_dict.values())
        print(
            f"🧩 Hetero graph: tracks={self.num_tracks}, artists={self.num_artists}, "
            f"genres={self.num_genres}, edges(total directed)={n_e}"
        )

    def get_graph_tensors(self) -> Tuple[Metadata, Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        return self.metadata, self.x_dict, self.edge_index_dict

    @property
    def num_nodes(self) -> int:
        """RL uses track nodes only; kept for logging parity with homogeneous builder."""
        return self.num_tracks


def build_hetero_music_graph(track_metadata_path: str) -> HeteroMusicGraph:
    b = MusicHeteroGraphBuilder(track_metadata_path)
    meta, xd, ed = b.get_graph_tensors()
    return HeteroMusicGraph(
        metadata=meta,
        x_dict=xd,
        edge_index_dict=ed,
        num_tracks=b.num_tracks,
        num_artists=b.num_artists,
        num_genres=b.num_genres,
    )


if __name__ == "__main__":
    path = "./data/hetrec2011-lastfm/track_metadata.json"
    g = build_hetero_music_graph(path)
    print("metadata node types:", g.metadata[0])
    print("edge types:", len(g.metadata[1]))
    for k, t in g.x_dict.items():
        print(f"  x[{k}]: {tuple(t.shape)}")
    for k, e in g.edge_index_dict.items():
        print(f"  edge{k}: {e.shape[1]} edges")
