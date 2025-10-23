import pandas as pd
import numpy as np

# 基本10曲データ
base_songs = [
    ("Shape of You", "Pop"),
    ("Bohemian Rhapsody", "Rock"),
    ("Lose Yourself", "Hip-Hop"),
    ("Levels", "Electronic"),
    ("Canon in D", "Classical"),
    ("Take Five", "Jazz"),
    ("Take Me Home", "Country"),
    ("No Woman No Cry", "Reggae"),
    ("The Thrill is Gone", "Blues"),
    ("Fluorescent Adolescent", "Indie")
]

genres = ["Pop","Rock","Hip-Hop","Electronic","Classical","Jazz","Country","Reggae","Blues","Indie"]

rows = []

for i in range(100):
    song_id = i + 1
    # 曲名は "タイトル + 番号"
    title = f"Song {song_id}"
    # 基本10曲からジャンルをランダム選択
    genre = np.random.choice(genres)
    # ランダム特徴量
    danceability = np.round(np.random.uniform(0.3, 0.9), 2)
    energy = np.round(np.random.uniform(0.3, 0.9), 2)
    tempo = np.random.randint(110, 150)
    valence = np.round(np.random.uniform(0.2, 0.8), 2)
    rows.append([song_id, title, danceability, energy, tempo, valence, genre])

df = pd.DataFrame(rows, columns=["song_id","title","danceability","energy","tempo","valence","genre"])
df.to_csv("songs_100.csv", index=False)
print("✅ songs_100.csv を作成しました")
