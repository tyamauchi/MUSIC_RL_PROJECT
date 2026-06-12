"""
グラフ構築ユーティリティ
track_metadata.jsonから音楽グラフを構築
"""

import torch
import json
import numpy as np
from collections import defaultdict


class MusicGraphBuilder:
    """音楽グラフ構築クラス"""
    
    def __init__(self, track_metadata_path):
        """
        初期化
        
        Args:
            track_metadata_path: track_metadata.jsonのパス
        """
        with open(track_metadata_path, 'r') as f:
            self.track_data = json.load(f)
        
        self.num_nodes = len(self.track_data)
        self.edge_index, self.node_features = self._build_graph()
        
        print(f"🔗 グラフ構築完了: {self.num_nodes}ノード, {self.edge_index.shape[1]}エッジ")
    
    def _build_graph(self):
        """ノードとエッジを構築"""
        edges = []
        node_features = []
        
        # ノード特徴量を準備
        for track_id, track_info in self.track_data.items():
            features = [
                track_info['popularity'] / 100.0,      # 人気度（正規化）
                track_info['duration_ms'] / 300000.0,  # 長さ（正規化）
                track_info['release_year'] / 2020.0     # 年代（正規化）
            ]
            node_features.append(features)
        
        # エッジ構築（同じジャンル）
        genre_to_tracks = defaultdict(list)
        for track_id, track_info in self.track_data.items():
            genre_to_tracks[track_info['genre']].append(int(track_id))
        
        print(f"🎵 ジャンル数: {len(genre_to_tracks)}")
        for genre, tracks in genre_to_tracks.items():
            print(f"  - {genre}: {len(tracks)}トラック")
            
            # 同じジャンル内で完全連結グラフを構築
            for i, track1 in enumerate(tracks):
                for track2 in tracks[i+1:]:
                    # 双方向エッジ
                    edges.append([track1, track2])
                    edges.append([track2, track1])
        
        # 追加エッジ構築（同じアーティスト）
        artist_to_tracks = defaultdict(list)
        for track_id, track_info in self.track_data.items():
            artist_to_tracks[track_info['artist']].append(int(track_id))
        
        print(f"🎤 アーティスト数: {len(artist_to_tracks)}")
        for artist, tracks in artist_to_tracks.items():
            if len(tracks) > 1:  # 複数トラックを持つアーティストのみ
                for i, track1 in enumerate(tracks):
                    for track2 in tracks[i+1:]:
                        # 同じアーティストのエッジ（重みを高くするために複数回追加）
                        edges.append([track1, track2])
                        edges.append([track2, track1])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        return edge_index, node_features
    
    def get_graph_data(self):
        """グラフデータを取得"""
        return self.edge_index, self.node_features
    
    def get_node_info(self, node_id):
        """ノード情報を取得"""
        if str(node_id) in self.track_data:
            return self.track_data[str(node_id)]
        return None
    
    def get_similar_nodes(self, node_id, similarity_type="genre"):
        """類似ノードを取得"""
        if str(node_id) not in self.track_data:
            return []
        
        target_track = self.track_data[str(node_id)]
        similar_nodes = []
        
        for track_id, track_info in self.track_data.items():
            if int(track_id) == node_id:
                continue
                
            if similarity_type == "genre" and track_info['genre'] == target_track['genre']:
                similar_nodes.append(int(track_id))
            elif similarity_type == "artist" and track_info['artist'] == target_track['artist']:
                similar_nodes.append(int(track_id))
        
        return similar_nodes


def create_music_graph(track_metadata_path):
    """
    音楽グラフを作成するヘルパー関数
    
    Args:
        track_metadata_path: track_metadata.jsonのパス
    
    Returns:
        edge_index: エッジインデックス
        node_features: ノード特徴量
        graph_builder: グラフビルダーインスタンス
    """
    graph_builder = MusicGraphBuilder(track_metadata_path)
    return graph_builder.get_graph_data(), graph_builder


if __name__ == "__main__":
    # テスト用
    track_metadata_path = "./data/hetrec2011-lastfm/track_metadata.json"
    edge_index, node_features, graph_builder = create_music_graph(track_metadata_path)
    
    print(f"グラフ情報:")
    print(f"  ノード数: {node_features.shape[0]}")
    print(f"  特徴量次元: {node_features.shape[1]}")
    print(f"  エッジ数: {edge_index.shape[1]}")
    print(f"  サンプルノード情報: {graph_builder.get_node_info(0)}")
