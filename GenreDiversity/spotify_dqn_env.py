# spotify_dqn_env.py
import numpy as np

class SimpleSpace:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n

    def sample(self):
        if self.n is not None:
            return np.random.randint(0, self.n)
        else:
            return np.random.rand(*self.shape)

class SpotifyPlaylistEnv:
    def __init__(self, song_features, user_profile, genres):
        self.song_features = song_features
        self.user_profile = user_profile
        self.genres = genres
        self.n_songs = len(song_features)
        self.current_index = 0
        self.done = False
        self.played_songs = []

        self.observation_space = SimpleSpace(shape=user_profile.shape)
        self.action_space = SimpleSpace(n=self.n_songs)

    def reset(self):
        self.current_index = 0
        self.done = False
        self.played_songs = []
        return self.user_profile

    def step(self, action):
        song = self.song_features[action]
        song_genre = self.genres[action]

        # 類似度報酬
        cosine_reward = np.dot(song, self.user_profile) / (
            np.linalg.norm(song) * np.linalg.norm(self.user_profile) + 1e-8
        )
        cosine_reward *= 2.0

        # 重複曲ペナルティ
        if action in self.played_songs:
            duplicate_count = self.played_songs.count(action)
            duplicate_penalty = -0.5 * (duplicate_count + 1)
        else:
            duplicate_penalty = 0.0

        # ジャンル多様性報酬
        diversity_reward = 0.0
        if len(self.played_songs) > 0:
            last_genre = self.genres[self.played_songs[-1]]
            if song_genre != last_genre:
                diversity_reward += 0.3
        unique_genres = len(set([self.genres[i] for i in self.played_songs] + [song_genre]))
        diversity_reward += 0.4 * (unique_genres / len(self.genres))

        reward = cosine_reward + duplicate_penalty + diversity_reward

        # 状態更新
        self.played_songs.append(action)
        self.current_index += 1
        if self.current_index >= min(self.n_songs, 10):
            self.done = True

        next_state = self.user_profile
        return next_state, reward, self.done, {}
