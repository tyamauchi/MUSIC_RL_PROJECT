import numpy as np
import os

# --------------------------------------------------
#  Multi-hot にするための次元数
#  → Last.fm の artistID の最大値 + 1 にする
# --------------------------------------------------
MAX_ARTIST_ID = 120000   # 必要なら自動推定に変更可


def to_multihot(ids, dim=MAX_ARTIST_ID):
    """与えられた ID リストを multi-hot ベクトルに変換"""
    vec = np.zeros(dim, dtype=np.float32)
    for x in ids:
        if 0 <= x < dim:
            vec[x] = 1.0
    return vec


def create_lastfm_trajectories(
    X_path="X_train.npy",
    y_path="y_train.npy",
    save_path="trajectories_lastfm.npy"
):
    # --------------------------------------------------
    #  Load X, y
    # --------------------------------------------------
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"{X_path} が見つかりません")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"{y_path} が見つかりません")

    print("Loading X and y...")
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded X: {X.shape}, y: {y.shape}")

    trajectories = []
    N = len(X)

    print("Converting to compact trajectories (store indices, not dense multi-hot)...")

    # --------------------------------------------------
    #  X[i] は 「過去10個の artistID」
    #  y[i] は 「次の artistID」
    #  LSTM の pretrain が期待する形式に合わせる:
    #    state_with_action: [track_history(20), response_history(20), action] (合計 41)
    #    target/response: float (0..1)
    #  response_history 情報は Last.fm 生データには無いので 0 埋めする。
    # --------------------------------------------------
    # Prepare artist pool from X (unique artist IDs present) to sample negatives
    try:
        artist_pool = np.unique(X.flatten()).astype(int)
    except Exception:
        artist_pool = np.arange(MAX_ARTIST_ID)

    for i in range(N):
        past_ids = list(map(int, X[i].tolist()))  # 長さ 10 を想定
        next_id = int(y[i])

        # track_history: past ids を左詰めで 20 長にし、残りは 0.0
        track_history = [float(x) for x in past_ids] + [0.0] * (20 - len(past_ids))

        # response_history: 実データに応答値がないため 0.0 で埋める
        response_history = [0.0] * 20

        current_state = track_history + response_history  # length 40

        # Positive example: next_id が再生された -> response = 1.0
        pos_state = np.array(current_state + [float(next_id)], dtype=np.float32)
        trajectories.append((pos_state, 1.0))

        # Negative example: ランダムな別トラックを action として response = 0.0
        # artist_pool から next_id を除外してサンプリング
        try:
            candidates = artist_pool[artist_pool != next_id]
            if len(candidates) == 0:
                raise ValueError
            neg_action = int(np.random.choice(candidates))
        except Exception:
            # フォールバック: 0..MAX_ARTIST_ID-1 からランダムに選ぶ（next_id を除外）
            neg_action = int(np.random.choice([x for x in range(MAX_ARTIST_ID) if x != next_id]))

        neg_state = np.array(current_state + [float(neg_action)], dtype=np.float32)
        trajectories.append((neg_state, 0.0))

    print(f"Generated {len(trajectories)} trajectories.")

    # --------------------------------------------------
    #  NumPy の object array として保存（各要素は小さいオブジェクトなのでメモリ効率良い）
    # --------------------------------------------------
    trajectories_array = np.array(trajectories, dtype=object)
    np.save(save_path, trajectories_array, allow_pickle=True)

    print(f"Saved to {save_path}")
    return trajectories_array


if __name__ == "__main__":
    create_lastfm_trajectories()
