import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from main_paperMD import LSTMUserSimulator, ActionHeadDQN, MusicEnvironment

# トラック情報の読み込み
def load_track_info(file_path=None):
    """トラック情報の読み込み。複数のパスを試行"""
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
            print(f"トラック情報を {path} から読み込み試行中...")
            tracks_df = pd.read_csv(path)
            # track_idとtitleのマッピングを作成
            track_titles = dict(zip(tracks_df['track_id'], tracks_df['title']))
            print(f"成功: {len(track_titles)} 件のトラック情報を読み込みました")
            return track_titles
        except FileNotFoundError:
            continue
    
    print("警告: トラック情報ファイルが見つかりません。トラックIDのみ表示します。")
    print("以下のパスを試行しました:")
    for path in possible_paths:
        print(f"- {path}")
    return {}

# デバイスの設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(model_path, state_dim=40, action_feature_dim=64):
    """学習済みモデルの読み込み"""
    # DQNモデルの読み込み
    policy_net = ActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
    policy_net.load_state_dict(torch.load(f'{model_path}/dqn_model.pth'))
    policy_net.eval()

    # User Simulatorの読み込み
    user_simulator = LSTMUserSimulator(state_dim + 1).to(DEVICE)
    user_simulator.load_state_dict(torch.load(f'{model_path}/user_simulator.pth'))
    user_simulator.eval()

    return policy_net, user_simulator

def generate_playlist(policy_net, user_simulator, track_pool_size=1000, session_length=20, 
                     state_dim=40, action_feature_dim=64, temperature=1.0):
    """プレイリストの生成"""
    env = MusicEnvironment(user_simulator, track_pool_size, session_length, state_dim)
    state = env.reset()
    
    # アクション特徴量の生成
    action_features = torch.randn(track_pool_size, action_feature_dim, device=DEVICE)
    
    playlist = []
    responses = []
    total_reward = 0
    
    with torch.no_grad():
        while True:
            # 状態をテンソルに変換
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # Q値の計算
            q_values = policy_net(state_tensor, action_features)
            
            # 既に選択されたトラックにペナルティを適用
            state_array = np.array(state)
            track_history = state_array[:len(state_array)//2]
            for i in range(action_features.size(0)):
                if i in track_history:
                    q_values[0, i] -= float('inf')  # 既存のトラックを除外
            
            # Softmax温度によるサンプリング
            q_values = q_values / temperature
            probs = torch.softmax(q_values, dim=1)
            action = torch.multinomial(probs[0], 1).item()
            
            # 環境でのステップ実行
            next_state, reward, done = env.step(action)
            
            # 結果の記録
            playlist.append(action)
            responses.append(env.response_history[env.current_step-1])
            total_reward += reward
            
            if done:
                break
            
            state = next_state
    
    return {
        'playlist': playlist,
        'responses': responses,
        'total_reward': total_reward,
        'average_response': np.mean(responses),
        'response_std': np.std(responses)
    }

def analyze_playlist(result, track_titles=None):
    """プレイリストの分析"""
    if track_titles is None:
        track_titles = {}
        
    print("\nプレイリスト生成結果の分析:")
    print(f"総合報酬: {result['total_reward']:.2f}")
    print(f"平均応答スコア: {result['average_response']:.3f}")
    print(f"応答の標準偏差: {result['response_std']:.3f}")
    print("\n各トラックの応答スコア:")
    for i, (track, response) in enumerate(zip(result['playlist'], result['responses']), 1):
        title = track_titles.get(track, "タイトル不明")
        print(f"トラック {i:2d}: ID {track:4d} - {title} - 応答スコア: {response:.3f}")

def main():
    # パラメータ設定
    model_path = 'saved_models'  # 学習済みモデルが保存されているディレクトリ
    state_dim = 40
    action_feature_dim = 64
    track_pool_size = 100
    session_length = state_dim // 2
    temperature = 0.5  # 生成の多様性を制御（低いほど決定論的）
    
    # トラック情報の読み込み
    track_titles = load_track_info()
    
    try:
        # モデルの読み込み
        policy_net, user_simulator = load_models(model_path, state_dim, action_feature_dim)
        
        print("プレイリストを生成中...")
        # プレイリストの生成
        result = generate_playlist(
            policy_net, 
            user_simulator,
            track_pool_size=track_pool_size,
            session_length=session_length,
            state_dim=state_dim,
            action_feature_dim=action_feature_dim,
            temperature=temperature
        )
        
        # 結果の分析と表示
        analyze_playlist(result, track_titles)
        
    except FileNotFoundError:
        print("エラー: 学習済みモデルが見つかりません。")
        print("先にモデルのトレーニングを実行し、以下のファイルを保存してください:")
        print(f"- {model_path}/dqn_model.pth")
        print(f"- {model_path}/user_simulator.pth")

if __name__ == "__main__":
    main()