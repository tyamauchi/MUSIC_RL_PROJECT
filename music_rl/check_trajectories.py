#!/usr/bin/env python3
"""
調査スクリプト: trajectories_lastfm.npy の構造とartist ID範囲を確認
"""

import numpy as np

LASTFM_TRAJ_PATH = "./data/hetrec2011-lastfm/trajectories_lastfm.npy"

print("Loading trajectories...")
trajectories = np.load(LASTFM_TRAJ_PATH, allow_pickle=True)

print(f"\n=== 基本情報 ===")
print(f"Total trajectories: {len(trajectories)}")
print(f"Type: {type(trajectories)}")

# サンプルを確認
sample = trajectories[0]
print(f"\n=== サンプル構造 ===")
print(f"Sample type: {type(sample)}")
print(f"Sample length: {len(sample)}")
print(f"Sample content: {sample}")

# 複数サンプルをチェック
print(f"\n=== 複数サンプル確認 ===")
for i in range(min(5, len(trajectories))):
    s = trajectories[i]
    print(f"Sample {i}: type={type(s)}, len={len(s) if hasattr(s, '__len__') else 'N/A'}, content={s}")
    
    # stateの構造を解析
    # state = [track_history(20), response_history(20), action] = 41要素
    if isinstance(s, (list, tuple)) and len(s) == 2:
        state, response = s
        if hasattr(state, '__len__') and len(state) == 41:
            track_history = state[:20]
            action = state[40]
            print(f"  -> State分解: track_history[:5]={list(track_history[:5])}, action={action}")

# 全軌跡からユニークなartist IDを抽出
print(f"\n=== Artist ID統計 ===")
all_artist_ids = set()

for i, traj in enumerate(trajectories):
    # trajはnumpy array [state_array, response] または list/tuple
    if hasattr(traj, '__len__') and len(traj) == 2:
        state = traj[0]
        if hasattr(state, '__len__') and len(state) == 41:
            # track_history (最初の20要素) と action (最後の要素) からIDを抽出
            # stateはnumpy array、要素はfloat
            track_history = state[:20]
            for track_id in track_history:
                if track_id > 0:  # 0はパディングと仮定
                    all_artist_ids.add(int(track_id))
            action_id = int(state[40])
            all_artist_ids.add(action_id)

print(f"Unique artist IDs count: {len(all_artist_ids)}")
print(f"Min artist ID: {min(all_artist_ids) if all_artist_ids else 'N/A'}")
print(f"Max artist ID: {max(all_artist_ids) if all_artist_ids else 'N/A'}")

# IDの分布を確認
sorted_ids = sorted(all_artist_ids)
print(f"\nFirst 10 IDs: {sorted_ids[:10]}")
print(f"Last 10 IDs: {sorted_ids[-10:]}")

# 統計
import statistics
if len(all_artist_ids) > 0:
    print(f"\nID range: {max(all_artist_ids) - min(all_artist_ids) + 1}")
