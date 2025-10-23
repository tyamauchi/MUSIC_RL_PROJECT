# main.py
import numpy as np
import torch
from spotify_dqn_env import SpotifyPlaylistEnv
from dqn_agent import train_dqn

# データ準備
n_songs = 100
feature_dim = 10

song_features = np.random.rand(n_songs, feature_dim)
user_profile = np.random.rand(feature_dim)

possible_genres = ["Pop", "Rock", "Jazz", "Hip-Hop", "Classical", "Electronic", "Reggae"]
genres = np.random.choice(possible_genres, size=n_songs)

# 環境作成
env = SpotifyPlaylistEnv(song_features, user_profile, genres)

# DQN学習
trained_net = train_dqn(env, episodes=200)

# モデル保存
torch.save(trained_net.state_dict(), "dqn_model.pth")
print("モデルを保存しました")
