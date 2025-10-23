# recommend_multi.py
import torch
import numpy as np
import pandas as pd
from spotify_dqn_env import SpotifyPlaylistEnv
from dqn_agent import DQN  # DQN用
from rl_playlist_agent import PolicyNet  # REINFORCE用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CSV読み込み
# -------------------------
df = pd.read_csv("songs_100.csv")
# 学習時次元に合わせる場合
features_rl = ['danceability','energy','tempo','valence']
song_features_rl = df[features_rl].values

# 推論用ゼロパディング版（10次元）
features_pad = ['danceability','energy','tempo','valence']
song_features_pad = df[features_pad].values
num_features = 10
if song_features_pad.shape[1] < num_features:
    pad = np.zeros((song_features_pad.shape[0], num_features - song_features_pad.shape[1]))
    song_features_pad = np.hstack([song_features_pad, pad])

# ユーザープロファイル（例として1曲目）
user_profile_rl = song_features_rl[0]
user_profile_pad = song_features_pad[0]

genres = df['genre'].tolist()

# 環境は10次元版を使用
env = SpotifyPlaylistEnv(song_features_pad, user_profile_pad, genres)

state_dim_pad = user_profile_pad.shape[0]
action_dim = env.n_songs

# -------------------------
# モデル選択
# -------------------------
model_type = input("使用モデルを選択 (dqn/reinforce): ").strip().lower()

if model_type == "dqn":
    net = DQN(state_dim_pad, action_dim).to(device)
    net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
    net.eval()
    
    # 推論
    state_tensor = torch.FloatTensor(user_profile_pad).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = net(state_tensor).squeeze(0)
    top_indices = torch.topk(q_values, 10).indices.cpu().numpy()
    
elif model_type == "reinforce":
    net = PolicyNet(state_dim=len(user_profile_rl), action_dim=action_dim).to(device)
    net.load_state_dict(torch.load("rl_playlist_policy_v2.pth", map_location=device))
    net.eval()
    
    # 推論
    state_tensor = torch.FloatTensor(user_profile_rl).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = net(state_tensor).squeeze(0)
    top_indices = torch.topk(probs, 10).indices.cpu().numpy()
    
else:
    raise ValueError("モデルタイプは dqn または reinforce のどちらかを指定してください")

# -------------------------
# 推薦曲表示
# -------------------------
recommended = []
for idx in top_indices:
    idx_mod = idx % len(df)
    if idx_mod not in recommended:
        recommended.append(idx_mod)
    if len(recommended) >= 10:
        break

for idx in recommended:
    row = df.iloc[idx]
    print(f"曲ID: {row['song_id']}, タイトル: {row['title']}, ジャンル: {row['genre']}, "
          f"danceability: {row['danceability']}, energy: {row['energy']}, "
          f"tempo: {row['tempo']}, valence: {row['valence']}")
