import torch
import numpy as np
import json
import os
from datetime import datetime
from config.config import DEVICE, STATE_DIM, ACTION_FEATURE_DIM, SESSION_LENGTH, TRACK_POOL_SIZE
from models.dqn import DuelingActionHeadDQN
from simulators.user_simulator import DeterministicUserSimulator
from agents.dqn_agent import DoubleDQNAgent

# 曲メタデータの読み込み（存在しない場合は自動生成）
TRACK_DB = None
try:
    from track_metadata import load_track_metadata, get_track_info
    TRACK_DB = load_track_metadata()
    print(f"✓ 曲メタデータ読み込み完了: {len(TRACK_DB)}曲")
except ImportError as e:
    print(f"⚠ track_metadata.pyが見つかりません: {e}")
    print("  Track IDのみ表示します。")
except Exception as e:
    print(f"⚠ メタデータ読み込みエラー: {e}")
    print("  Track IDのみ表示します。")


class PlaylistRecommender:
    def __init__(self, model_dir='saved_models'):
        """
        学習済みモデルを読み込んで推薦を行うクラス
        
        Args:
            model_dir: モデルが保存されているディレクトリ
        """
        self.model_dir = self._find_latest_model(model_dir)
        print(f"モデル読み込み: {self.model_dir}")
        
        # エージェントの初期化と読み込み
        self.agent = DoubleDQNAgent(STATE_DIM, ACTION_FEATURE_DIM, use_dueling=True)
        model_path = os.path.join(self.model_dir, 'policy_net.pth')
        self.agent.load_model(model_path)
        
        # アクション特徴量の読み込み
        action_features_path = os.path.join(self.model_dir, 'action_features.pth')
        self.action_features = torch.load(action_features_path, map_location=DEVICE)
        
        # メトリクスの読み込み
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            
            # メトリクスの表示（キーが存在する場合のみ）
            if 'avg_reward' in self.metrics:
                print(f"✓ モデル性能: 平均報酬 {self.metrics['avg_reward']:.2f}")
            else:
                print(f"✓ メトリクス: {self.metrics}")
        else:
            print("⚠ metrics.jsonが見つかりません")
            self.metrics = {}
        
        print(f"✓ アクション数: {self.action_features.size(0)} tracks")
    
    def _find_latest_model(self, base_dir):
        """最新のモデルディレクトリを見つける"""
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {base_dir}")
        
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not model_dirs:
            raise FileNotFoundError(f"モデルが見つかりません: {base_dir}")
        
        # 最新のディレクトリを選択
        latest_dir = sorted(model_dirs)[-1]
        return os.path.join(base_dir, latest_dir)
    
    def generate_playlist(self, session_length=SESSION_LENGTH, avoid_duplicates=True, diversity_bonus=False):
        """
        プレイリストを生成
        
        Args:
            session_length: プレイリストの長さ
            avoid_duplicates: 重複を避けるかどうか
            diversity_bonus: 多様性ボーナスを適用するかどうか
        
        Returns:
            playlist: 推薦された曲のIDリスト
            q_values_history: 各ステップのQ値
        """
        state = [0.0] * STATE_DIM
        playlist = []
        q_values_history = []
        
        for step in range(session_length):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                q_values = self.agent.policy_net(state_tensor, self.action_features)
                
                # 重複ペナルティ
                if avoid_duplicates:
                    penalties = torch.zeros_like(q_values)
                    track_history = state[:STATE_DIM//2]
                    for i in range(self.action_features.size(0)):
                        if i in track_history:
                            penalties[0, i] = -10.0  # 強いペナルティ
                    q_values = q_values + penalties
                
                # 多様性ボーナス（オプション）
                if diversity_bonus and len(playlist) > 0:
                    # 最近選んだ曲と異なる特徴を持つ曲を優遇
                    recent_features = self.action_features[playlist[-1]].unsqueeze(0)
                    similarities = torch.nn.functional.cosine_similarity(
                        self.action_features, recent_features, dim=1
                    )
                    diversity_bonus_values = (1 - similarities) * 0.5
                    q_values[0] += diversity_bonus_values
                
                # 最良のアクションを選択
                action = torch.argmax(q_values).item()
                q_values_history.append(q_values[0].cpu().numpy())
            
            # 状態を更新（実際の応答は不明なので0.8と仮定）
            playlist.append(action)
            state[step] = float(action)
            state[STATE_DIM//2 + step] = 0.8  # 仮の応答値
        
        return playlist, q_values_history
    
    def generate_multiple_playlists(self, n_playlists=5, **kwargs):
        """
        複数のプレイリストを生成
        
        Args:
            n_playlists: 生成するプレイリスト数
            **kwargs: generate_playlistに渡す引数
        
        Returns:
            playlists: プレイリストのリスト
        """
        playlists = []
        for i in range(n_playlists):
            playlist, _ = self.generate_playlist(**kwargs)
            playlists.append(playlist)
        return playlists
    
    def display_playlist(self, playlist, title="推薦プレイリスト", show_details=True):
        """
        プレイリストを表示
        
        Args:
            playlist: 曲IDのリスト
            title: プレイリストのタイトル
            show_details: 詳細情報を表示するかどうか
        """
        print("\n" + "=" * 70)
        print(f"{title}")
        print("=" * 70)
        
        if TRACK_DB and show_details:
            # 詳細表示
            for i, track_id in enumerate(playlist, 1):
                info = get_track_info(track_id, TRACK_DB)
                print(f"  {i:2d}. {info['name']:<30} - {info['artist']:<25}")
                print(f"      [{info['genre']:<12}] {info['duration_str']:<6} (人気度: {info['popularity']})")
        else:
            # シンプル表示
            for i, track_id in enumerate(playlist, 1):
                print(f"  {i:2d}. Track #{track_id:3d}")
        
        print("=" * 70 + "\n")
    
    def evaluate_with_simulator(self, playlist, deterministic_sim=None):
        """
        決定論的シミュレータでプレイリストを評価
        
        Args:
            playlist: 評価する曲IDのリスト
            deterministic_sim: ユーザーシミュレータ（Noneの場合は新規作成）
        
        Returns:
            evaluation: 評価結果の辞書
        """
        if deterministic_sim is None:
            deterministic_sim = DeterministicUserSimulator(TRACK_POOL_SIZE)
        
        deterministic_sim.reset_user_state()
        
        state = [0.0] * STATE_DIM
        responses = []
        rewards = []
        
        for step, track_id in enumerate(playlist):
            response = deterministic_sim.get_response(state, track_id, step)
            responses.append(response)
            
            # 報酬計算（環境と同じロジック）
            base_reward = response
            if track_id in playlist[:step]:
                base_reward *= 0.2
            
            rewards.append(base_reward)
            
            # 状態更新
            state[step] = float(track_id)
            state[STATE_DIM//2 + step] = response
        
        # セッション終了ボーナス
        avg_response = np.mean(responses)
        if avg_response > 0.9:
            rewards[-1] += 2.0
        elif avg_response > 0.8:
            rewards[-1] += 1.0
        
        evaluation = {
            'total_reward': sum(rewards),
            'avg_response': avg_response,
            'min_response': min(responses),
            'max_response': max(responses),
            'unique_tracks': len(set(playlist)),
            'duplicate_rate': 1.0 - len(set(playlist)) / len(playlist)
        }
        
        return evaluation, responses
    
    def save_playlist(self, playlist, filename=None, output_dir='playlists'):
        """
        プレイリストをJSONファイルに保存
        
        Args:
            playlist: 曲IDのリスト
            filename: 保存するファイル名（Noneの場合は自動生成）
            output_dir: 保存先ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'playlist_{timestamp}.json'
        
        filepath = os.path.join(output_dir, filename)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'model_dir': self.model_dir,
            'playlist': playlist,
            'length': len(playlist),
            'unique_tracks': len(set(playlist))
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ プレイリスト保存: {filepath}")


def main():
    """メイン実行関数"""
    print("\n" + "=" * 60)
    print("🎵 音楽推薦システム - プレイリスト生成")
    print("=" * 60 + "\n")
    
    # 推薦システムの初期化
    recommender = PlaylistRecommender()
    
    # シミュレータの準備（評価用）
    simulator = DeterministicUserSimulator(TRACK_POOL_SIZE)
    
    # プレイリスト生成
    print("\n【オプション1】標準プレイリスト（重複なし）")
    playlist1, _ = recommender.generate_playlist(avoid_duplicates=True, diversity_bonus=False)
    recommender.display_playlist(playlist1, title="標準プレイリスト")
    
    eval1, responses1 = recommender.evaluate_with_simulator(playlist1, simulator)
    print(f"評価:")
    print(f"  総報酬: {eval1['total_reward']:.2f}")
    print(f"  平均応答率: {eval1['avg_response']:.3f}")
    print(f"  重複率: {eval1['duplicate_rate']:.1%}")
    
    # 多様性重視プレイリスト
    print("\n【オプション2】多様性重視プレイリスト")
    playlist2, _ = recommender.generate_playlist(avoid_duplicates=True, diversity_bonus=True)
    recommender.display_playlist(playlist2, title="多様性重視プレイリスト")
    
    eval2, responses2 = recommender.evaluate_with_simulator(playlist2, simulator)
    print(f"評価:")
    print(f"  総報酬: {eval2['total_reward']:.2f}")
    print(f"  平均応答率: {eval2['avg_response']:.3f}")
    print(f"  重複率: {eval2['duplicate_rate']:.1%}")
    
    # 複数プレイリスト生成
    print("\n【オプション3】複数プレイリスト生成")
    playlists = recommender.generate_multiple_playlists(n_playlists=3)
    for i, playlist in enumerate(playlists, 1):
        recommender.display_playlist(playlist, title=f"プレイリスト #{i}")
    
    # プレイリストを保存
    print("\n【保存】")
    recommender.save_playlist(playlist1, filename='standard_playlist.json')
    recommender.save_playlist(playlist2, filename='diverse_playlist.json')
    
    print("\n" + "=" * 60)
    print("✓ 完了")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()