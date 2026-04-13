import pandas as pd
import numpy as np
from collections import defaultdict

# ファイルパス
DATA_PATH = "./hetrec2011-lastfm-2k"

# =============================
# 1. データ読み込み
# =============================
user_artists = pd.read_csv(
    f"{DATA_PATH}/user_artists.dat",
    sep="\t"
)

print("Loaded:", user_artists.head())

# =============================
# 2. ユーザーごとに artistID のリストを作る
#    再生回数(weight)でソートして疑似時系列化
# =============================

user_sequences = defaultdict(list)

for user, group in user_artists.groupby("userID"):
    # 再生回数順で並べる（多い=よく聞く）
    seq = group.sort_values("weight", ascending=False)["artistID"].tolist()
    user_sequences[user] = seq

print("Example user sequence:", list(user_sequences.items())[:1])

# =============================
# 3. 系列 → (input → next) に変換
# =============================

def generate_training_samples(sequences, seq_len=10):
    X, y = [], []

    for user, seq in sequences.items():
        if len(seq) <= seq_len:
            continue

        # スライドして系列を作る
        for i in range(len(seq) - seq_len):
            X.append(seq[i : i + seq_len])
            y.append(seq[i + seq_len])

    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = generate_training_samples(user_sequences, seq_len=SEQ_LEN)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =============================
# 保存したい場合
# =============================
np.save("X_train.npy", X)
np.save("y_train.npy", y)
print("Saved X_train.npy & y_train.npy")
