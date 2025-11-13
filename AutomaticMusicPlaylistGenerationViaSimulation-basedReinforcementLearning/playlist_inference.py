import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LSTM_HIDDEN_SIZE = [256, 128, 64]

class LSTMUserSimulator(nn.Module):
    """軽量版LSTMシミュレータ"""
    def __init__(self, input_size, hidden_sizes=LSTM_HIDDEN_SIZE):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[2], 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)

class DuelingActionHeadDQN(nn.Module):
    """Dueling Architecture + Action Head DQN"""
    def __init__(self, state_dim, action_feature_dim):
        super(DuelingActionHeadDQN, self).__init__()
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
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
        
        value = self.value_stream(state_features)
        num_actions = action_features.size(1)
        value = value.expand(-1, num_actions)
        
        combined = torch.cat([state_features_expanded, action_features_processed], dim=-1)
        advantage = self.advantage_stream(combined.view(-1, combined.size(-1))).view(batch_size, -1)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class MusicEnvironment:
    """音楽推薦環境（推論用）"""
    def __init__(self, user_simulator, track_pool_size=1000, session_length=20, state_dim=40):
        self.user_simulator = user_simulator
        self.track_pool_size = track_pool_size
        self.session_length = session_length
        self.state_dim = state_dim
        self.current_step = 0
        self.track_history = []
        self.response_history = []
        
    def reset(self):
        self.current_step = 0
        self.track_history = [0.0] * (self.state_dim // 2)
        self.response_history = [0.0] * (self.state_dim // 2)
        return self._get_state()

    def step(self, action):
        state = self._get_state()
        state_tensor = torch.FloatTensor([state + [float(action)]]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            response_prob, _ = self.user_simulator(state_tensor)
        response = float(response_prob.squeeze())
        
        self.track_history[self.current_step] = float(action)
        self.response_history[self.current_step] = response
        self.current_step += 1
        
        base_reward = response
        
        if float(action) in self.track_history[:self.current_step-1]:
            base_reward *= 0.2
        
        if self.current_step >= self.session_length:
            responses = np.array(self.response_history)
            avg_response = np.mean(responses)
            
            if avg_response > 0.9:
                base_reward += 2.0
            elif avg_response > 0.8:
                base_reward += 1.0
        
        done = self.current_step >= self.session_length
        return self._get_state(), base_reward, done

    def _get_state(self):
        return self.track_history + self.response_history

def load_track_info(file_path=None):
    """トラック情報の読み込み（カラム名の自動検出）"""
    possible_paths = [
        'data/tracks.csv',
        'tracks.csv',
        'data/track_data.csv',
        '../data/tracks.csv'
    ]
    
    if file_path:
        possible_paths.insert(0, file_path)
    
    for path in possible_paths:
        try:
            print(f" トラック情報を {path} から読み込み試行中...")
            tracks_df = pd.read_csv(path)
            
            # カラム名を確認
            print(f" カラム名: {list(tracks_df.columns)}")
            
            # track_idカラムの検出
            track_id_col = None
            for col in ['track_id', 'id', 'trackId', 'track_no']:
                if col in tracks_df.columns:
                    track_id_col = col
                    break
            
            if track_id_col is None:
                print(f" ⚠️ track_id カラムが見つかりません。最初のカラムを使用します")
                track_id_col = tracks_df.columns[0]
            
            # titleカラムの検出
            title_col = None
            for col in ['title', 'name', 'track_name', 'song_name', 'song']:
                if col in tracks_df.columns:
                    title_col = col
                    break
            
            if title_col is None:
                print(f" ⚠️ title カラムが見つかりません")
                # track_idをそのまま表示
                track_titles = {int(tid): f"Track {tid}" for tid in tracks_df[track_id_col]}
            else:
                # 正常にタイトルを読み込み
                track_titles = {}
                for _, row in tracks_df.iterrows():
                    tid = int(row[track_id_col])
                    title = row[title_col]
                    # NaNや空文字列の処理
                    if pd.isna(title) or str(title).strip() == '':
                        track_titles[tid] = f"Track {tid}"
                    else:
                        track_titles[tid] = str(title)
            
            print(f" 成功: {len(track_titles)} 件のトラック情報を読み込みました")
            print(f"   使用カラム: track_id={track_id_col}, title={title_col}")
            
            # サンプル表示
            print(f"\n サンプル（最初の5件）:")
            for i, (tid, title) in enumerate(list(track_titles.items())[:5]):
                print(f"   ID {tid:4d}: {title}")
            print()
            
            return track_titles, tracks_df
            
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(" ⚠️ トラック情報ファイルが見つかりません。ダミーデータを使用します。")
    # ダミーデータを生成
    dummy_titles = {i: f"Track {i}" for i in range(1000)}
    return dummy_titles, None

def load_models(model_path, state_dim=40, action_feature_dim=64):
    """学習済みモデルの読み込み（Action Features含む）"""
    print(f"📂 モデルを {model_path} から読み込み中...")
    
    # DQNモデルの読み込み
    policy_net = DuelingActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
    policy_net.load_state_dict(torch.load(f'{model_path}/policy_net.pth', map_location=DEVICE))
    policy_net.eval()
    print("✅ DQNモデルを読み込みました")

    # Action Featuresの読み込み
    action_features = None
    try:
        action_features = torch.load(f'{model_path}/action_features.pth', map_location=DEVICE)
        print(f"✅ Action Features を読み込みました (shape: {action_features.shape})")
    except FileNotFoundError:
        print("⚠️  Action Features が見つかりません。ランダムに初期化します。")
        torch.manual_seed(42)
        action_features = torch.randn(1000, action_feature_dim, device=DEVICE)

    # User Simulatorの読み込み
    user_simulator = None
    try:
        user_simulator = LSTMUserSimulator(state_dim + 1).to(DEVICE)
        user_simulator.load_state_dict(torch.load(f'{model_path}/user_simulator.pth', map_location=DEVICE))
        user_simulator.eval()
        print("✅ User Simulatorを読み込みました")
    except FileNotFoundError:
        print("⚠️  User Simulatorが見つかりません（オプション）")
    
    # メトリクスの読み込み
    try:
        with open(f'{model_path}/metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"✅ メトリクスを読み込みました")
        print(f"   - アーキテクチャ: {metrics.get('architecture', 'N/A')}")
        print(f"   - 最良報酬: {metrics.get('best_avg_reward', 'N/A'):.2f}")
        if 'test_results' in metrics:
            print(f"   - テスト応答率: {metrics['test_results'].get('avg_response', 'N/A'):.3f}")
    except FileNotFoundError:
        metrics = None
        print("⚠️  メトリクスファイルが見つかりません")
    except Exception as e:
        metrics = None
        print(f"⚠️  メトリクス読み込みエラー: {e}")
    
    print()
    return policy_net, user_simulator, metrics, action_features

def generate_playlist(policy_net, user_simulator, action_features, 
                     track_pool_size=1000, session_length=20, 
                     state_dim=40, temperature=0.3, seed=None):
    """プレイリストの生成"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    env = MusicEnvironment(user_simulator, track_pool_size, session_length, state_dim)
    state = env.reset()
    
    playlist = []
    responses = []
    q_values_history = []
    total_reward = 0
    
    with torch.no_grad():
        for step in range(session_length):
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # Q値の計算
            q_values = policy_net(state_tensor, action_features)
            
            # 既に選択されたトラックにペナルティ
            state_array = np.array(state)
            track_history = state_array[:len(state_array)//2]
            penalties = torch.zeros_like(q_values)
            for i in range(action_features.size(0)):
                if i in track_history:
                    penalties[0, i] = -1e10
            
            q_values_adjusted = q_values + penalties
            
            # Temperature scalingでサンプリング
            if temperature > 0:
                q_values_scaled = q_values_adjusted / temperature
                probs = torch.softmax(q_values_scaled, dim=1)
                action = torch.multinomial(probs[0], 1).item()
            else:
                action = torch.argmax(q_values_adjusted).item()
            
            # 環境でステップ実行
            next_state, reward, done = env.step(action)
            
            # 結果の記録
            playlist.append(int(action))
            responses.append(float(env.response_history[env.current_step-1]))
            q_values_history.append(float(q_values[0, action].item()))
            total_reward += reward
            
            state = next_state
    
    return {
        'playlist': playlist,
        'responses': responses,
        'q_values': q_values_history,
        'total_reward': total_reward,
        'average_response': np.mean(responses),
        'response_std': np.std(responses),
        'min_response': np.min(responses),
        'max_response': np.max(responses)
    }

def analyze_playlist(result, track_titles=None, tracks_df=None, save_path=None):
    """プレイリストの詳細分析"""
    
    print("\n" + "=" * 80)
    print("🎵 プレイリスト生成結果の分析")
    print("=" * 80)
    
    # 統計情報
    print(f"\n📊 統計情報:")
    print(f"  総合報酬:        {result['total_reward']:6.2f}")
    print(f"  平均応答スコア:  {result['average_response']:.3f}")
    print(f"  応答の標準偏差:  {result['response_std']:.3f}")
    print(f"  最小応答スコア:  {result['min_response']:.3f}")
    print(f"  最大応答スコア:  {result['max_response']:.3f}")
    
    # 品質評価
    high_quality = sum(1 for r in result['responses'] if r > 0.95)
    good_quality = sum(1 for r in result['responses'] if 0.9 < r <= 0.95)
    medium_quality = sum(1 for r in result['responses'] if 0.8 < r <= 0.9)
    
    print(f"\n🎯 品質分布:")
    print(f"  高品質 (>0.95):  {high_quality:2d} 曲 ({high_quality/len(result['responses'])*100:.1f}%)")
    print(f"  良品質 (0.9-0.95): {good_quality:2d} 曲 ({good_quality/len(result['responses'])*100:.1f}%)")
    print(f"  中品質 (0.8-0.9):  {medium_quality:2d} 曲 ({medium_quality/len(result['responses'])*100:.1f}%)")
    
    # トラックリスト
    print(f"\n🎼 プレイリスト詳細:")
    print("-" * 80)
    
    for i, (track_id, response, q_value) in enumerate(
        zip(result['playlist'], result['responses'], result['q_values']), 1
    ):
        # トラック情報の取得
        if track_titles and track_id in track_titles:
            title = track_titles[track_id]
        else:
            title = f"Track {track_id}"
        
        # 追加情報
        extra_info = ""
        if tracks_df is not None:
            # track_idカラムの検出
            id_col = None
            for col in ['track_id', 'id', 'trackId']:
                if col in tracks_df.columns:
                    id_col = col
                    break
            
            if id_col:
                track_row = tracks_df[tracks_df[id_col] == track_id]
                if not track_row.empty:
                    if 'artist' in track_row.columns:
                        artist = track_row['artist'].values[0]
                        if pd.notna(artist) and str(artist).strip():
                            extra_info = f" - {artist}"
        
        # 応答スコアによる評価マーク
        if response > 0.95:
            mark = "🌟"
        elif response > 0.9:
            mark = "⭐"
        elif response > 0.8:
            mark = "✨"
        else:
            mark = "  "
        
        print(f"{mark} {i:2d}. [ID:{track_id:4d}] {title:50s}{extra_info}")
        print(f"      応答: {response:.3f} | Q値: {q_value:7.3f}")
    
    print("-" * 80)
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'playlist': result['playlist'],
                'responses': result['responses'],
                'q_values': result['q_values'],
                'statistics': {
                    'total_reward': result['total_reward'],
                    'average_response': result['average_response'],
                    'response_std': result['response_std']
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 プレイリストを保存: {save_path}")

def generate_multiple_playlists(policy_net, user_simulator, action_features, 
                                n_playlists=5, **kwargs):
    """複数のプレイリストを生成して比較"""
    print(f"\n🔄 {n_playlists}個のプレイリストを生成中...\n")
    
    results = []
    for i in range(n_playlists):
        print(f"プレイリスト {i+1}/{n_playlists} 生成中...")
        result = generate_playlist(
            policy_net, user_simulator, action_features,
            seed=42 + i,
            **kwargs
        )
        results.append(result)
    
    # 比較統計
    print("\n📊 生成されたプレイリストの比較:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"プレイリスト {i}: 平均応答 {result['average_response']:.3f} "
              f"(範囲: {result['min_response']:.3f}-{result['max_response']:.3f})")
    
    # 最良のプレイリストを選択
    best_idx = np.argmax([r['average_response'] for r in results])
    print(f"\n🏆 最良のプレイリスト: #{best_idx + 1}")
    
    return results, best_idx

def main():
    print("\n" + "=" * 80)
    print("🎵 音楽プレイリスト生成システム（推論モード）")
    print("=" * 80 + "\n")
    
    # パラメータ設定
    model_path = 'saved_models/20251113_100051'  # 新しく学習したモデル
    state_dim = 40
    action_feature_dim = 64
    track_pool_size = 265  # 実際のCSVのトラック数に合わせる
    session_length = 20
    temperature = 0.3
    n_playlists = 3
    
    # トラック情報の読み込み（自動検出）
    track_titles, tracks_df = load_track_info()
    
    try:
        # モデルとAction Featuresの読み込み
        policy_net, user_simulator, metrics, action_features = load_models(
            model_path, state_dim, action_feature_dim
        )
        
        if action_features is None:
            print("⚠️  Action Featuresが見つからないため、推論精度が低下する可能性があります")
            return
        
        # 複数のプレイリストを生成
        results, best_idx = generate_multiple_playlists(
            policy_net,
            user_simulator,
            action_features,
            n_playlists=n_playlists,
            track_pool_size=track_pool_size,
            session_length=session_length,
            state_dim=state_dim,
            temperature=temperature
        )
        
        # 最良のプレイリストを詳細分析
        analyze_playlist(
            results[best_idx],
            track_titles,
            tracks_df,
            save_path=f'generated_playlists/playlist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}")
        print("\n必要なファイル:")
        print(f"  - {model_path}/policy_net.pth")
        print(f"  - {model_path}/action_features.pth (必須)")
        print(f"  - {model_path}/user_simulator.pth (オプション)")
        print(f"  - {model_path}/metrics.json (オプション)")

if __name__ == "__main__":
    main()