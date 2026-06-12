import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
from collections import deque

from config.config import *
from simulators.user_simulator import DeterministicUserSimulator, LSTMUserSimulator
from agents.dqn_agent import DoubleDQNAgent
from agents.ppo_agent import PPOAgent
from agents.simple_graph_dqn_agent import SimpleGraphDQNAgent
from agents.hetero_graph_dqn_agent import HeteroGraphDQNAgent
from environment.music_env import MusicEnvironment
from memory.replay_buffer import ReplayBuffer
from utils.training import generate_expert_trajectories, pretrain_lstm_with_expert_data
from utils.evaluation import evaluate_agent


# ======================================================
#  モード切り替えフラグ
# ======================================================

USE_REAL_LASTFM_DATA = False  # ← True: Last.fm 実データ / False: det_sim データ
LASTFM_TRAJ_PATH = "./data/hetrec2011-lastfm/trajectories_lastfm.npy"

# ======================================================
#  エージェント切り替え
# ======================================================
#AGENT_TYPE = "dqn"  # ← "dqn", "ppo", "graph_dqn", "hetero_graph_dqn"
#AGENT_TYPE = "ppo"  # ← "dqn", "ppo", "graph_dqn", "hetero_graph_dqn"
AGENT_TYPE = "hetero_graph_dqn"  # ← "dqn", "ppo", "graph_dqn", "hetero_graph_dqn"


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_{timestamp}'
    save_dir = f'saved_models/{timestamp}'
    writer = SummaryWriter(log_dir)
    
    print("\n" + "=" * 60)
    
    # エージェント名の設定
    if AGENT_TYPE == "ppo":
        agent_name = "PPO"
    elif AGENT_TYPE == "graph_dqn":
        agent_name = "Simple Graph DQN"
    elif AGENT_TYPE == "hetero_graph_dqn":
        agent_name = "Heterogeneous Graph DQN (HGT)"
    else:
        agent_name = "Double DQN + Dueling DQN"
    
    print(f"{agent_name} 音楽推薦システム")
    print("Mode:", "REAL Last.fm Data" if USE_REAL_LASTFM_DATA else "Deterministic Simulator")
    print("Agent:", AGENT_TYPE.upper())
    print("=" * 60 + "\n")
    
    # TensorBoardに設定情報を保存（Markdown形式）
    config_text = f"""### 実行設定

| 項目 | 値 |
|------|-----|
| **Agent Type** | {AGENT_TYPE.upper()} |
| **Mode** | {'REAL Last.fm Data (Top-2048)' if USE_REAL_LASTFM_DATA else 'Deterministic Simulator'} |
| **Agent Name** | {agent_name} |
| **Timestamp** | {timestamp} |
| **Random Seed** | {RANDOM_SEED} |
| **Track Pool Size** | {'2048 (Last.fm Top-N)' if USE_REAL_LASTFM_DATA else str(TRACK_POOL_SIZE)} |
| **Action Feature Dim** | {'2048 (One-hot)' if USE_REAL_LASTFM_DATA else str(ACTION_FEATURE_DIM)} |
"""
    writer.add_text('Config/Settings', config_text, 0)
    

    # ======================================================
    #  Trajectory 生成 / 読み込み
    # ======================================================
    if USE_REAL_LASTFM_DATA:
        print(">>> REAL Last.fm user data を使用します")
        if not os.path.exists(LASTFM_TRAJ_PATH):
            raise FileNotFoundError(f"{LASTFM_TRAJ_PATH} が見つかりません。先に Last.fm から生成してください。")
        trajectories = np.load(LASTFM_TRAJ_PATH, allow_pickle=True)
        print(f"Loaded {len(trajectories)} Last.fm trajectories")
        
        # ======================================================
        # 案C: Top-256頻出アーティスト絞込み
        # ======================================================
        print("\n--- Top-256 Artist Selection ---")
        from collections import Counter
        artist_counter = Counter()
        
        for traj in trajectories:
            if hasattr(traj, '__len__') and len(traj) == 2:
                state = traj[0]
                if hasattr(state, '__len__') and len(state) == 41:
                    # track_history (20要素) と action (1要素) からID抽出・カウント
                    for track_id in state[:20]:
                        if track_id > 0:
                            artist_counter[int(track_id)] += 1
                    artist_counter[int(state[40])] += 1
        
        # 頻出上位2048アーティストを選択
        TOP_N_ARTISTS = 2048
        top_n_artists = [aid for aid, _ in artist_counter.most_common(TOP_N_ARTISTS)]
        sorted_artists = sorted(top_n_artists)  # ソートして連続インデックス化
        artist_to_idx = {aid: i for i, aid in enumerate(sorted_artists)}
        idx_to_artist = {i: aid for aid, i in artist_to_idx.items()}
        n_unique_artists = len(sorted_artists)
        
        # カバレッジ統計
        total_appearances = sum(artist_counter.values())
        top_n_appearances = sum(artist_counter[aid] for aid in top_n_artists)
        coverage = top_n_appearances / total_appearances * 100
        
        print(f"Selected top {n_unique_artists} artists from {len(artist_counter)} unique")
        print(f"Coverage: {coverage:.1f}% of all appearances")
        print(f"--- Top-{TOP_N_ARTISTS} Selection完了 ---\n")

    else:
        print(">>> DeterministicUserSimulator を使用します")
        det_sim = DeterministicUserSimulator(TRACK_POOL_SIZE)
        trajectories = generate_expert_trajectories(
            det_sim,
            n_trajectories=N_TRAJECTORIES,
            session_length=SESSION_LENGTH,
            add_exploration=True
        )
        print(f"Generated {len(trajectories)} trajectories (det_sim)")


    # ======================================================
    #  LSTM UserSimulator の Pretrain
    # ======================================================
    lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
    print("\n--- LSTMUserSimulator PRETRAIN START ---\n")
    pretrain_lstm_with_expert_data(lstm_sim, trajectories, n_epochs=PRETRAIN_EPOCHS)
    print("\n--- LSTMUserSimulator PRETRAIN DONE ---\n")
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(lstm_sim.state_dict(), os.path.join(save_dir, 'user_simulator.pth'))
    

    # ======================================================
    #  RL Environment 構築
    # ======================================================
    # Last.fm使用時は動的にTRACK_POOL_SIZEを設定
    if USE_REAL_LASTFM_DATA:
        track_pool_size = n_unique_artists
    else:
        track_pool_size = TRACK_POOL_SIZE
    
    env = MusicEnvironment(lstm_sim, track_pool_size, SESSION_LENGTH, STATE_DIM)
    
    # エージェント初期化
    # Last.fm使用時は動的にACTION_FEATURE_DIMを設定（One-hotサイズ）
    if USE_REAL_LASTFM_DATA:
        action_feature_dim = n_unique_artists
    else:
        action_feature_dim = ACTION_FEATURE_DIM
    
    if AGENT_TYPE == "ppo":
        agent = PPOAgent(STATE_DIM, action_feature_dim)
    elif AGENT_TYPE == "graph_dqn":
        # Simple Graph DQNは両データ形式に対応
        track_metadata_path = "./data/hetrec2011-lastfm/track_metadata.json"
        data_source = "lastfm" if USE_REAL_LASTFM_DATA else "det_sim"
        agent = SimpleGraphDQNAgent(
            state_dim=STATE_DIM,
            track_metadata_path=track_metadata_path,
            data_source=data_source
        )
        agent.memory = deque(maxlen=MEMORY_SIZE)
        print(f"🚀 Simple Graph DQN 初期化完了 (データソース: {data_source})")
    elif AGENT_TYPE == "hetero_graph_dqn":
        track_metadata_path = "./data/hetrec2011-lastfm/track_metadata.json"
        data_source = "lastfm" if USE_REAL_LASTFM_DATA else "det_sim"
        agent = HeteroGraphDQNAgent(
            state_dim=STATE_DIM,
            track_metadata_path=track_metadata_path,
            data_source=data_source,
        )
        agent.memory = deque(maxlen=MEMORY_SIZE)
        print(f"🧩 Hetero Graph DQN 初期化完了 (データソース: {data_source})")
    else:
        agent = DoubleDQNAgent(STATE_DIM, action_feature_dim, use_dueling=True)
        agent.memory = ReplayBuffer()
    
    best_avg_reward = float('-inf')
    rewards_history = []
    
    torch.manual_seed(RANDOM_SEED)
    # One-hotエンコーディング: 各トラックを独立に識別可能に
    if USE_REAL_LASTFM_DATA:
        # 案D: マッピング後のインデックスでOne-hot作成
        cached_action_features = torch.eye(n_unique_artists, device=DEVICE)
        print(f"One-hot encoded: [{n_unique_artists}, {n_unique_artists}] (Last.fm)")
    else:
        # Deterministic Simulator: 固定サイズ
        cached_action_features = torch.eye(TRACK_POOL_SIZE, device=DEVICE)
        print(f"One-hot encoded: [{TRACK_POOL_SIZE}, {TRACK_POOL_SIZE}] (Simulator)")
    
    print("学習開始\n")
    

    # ======================================================
    #  RL 学習ループ
    # ======================================================
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        
        if AGENT_TYPE == "dqn" or AGENT_TYPE == "graph_dqn" or AGENT_TYPE == "hetero_graph_dqn":
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
        
        for step in range(SESSION_LENGTH):
            if AGENT_TYPE == "ppo":
                action, log_prob = agent.select_action_with_logprob(state, cached_action_features)
            elif AGENT_TYPE == "graph_dqn" or AGENT_TYPE == "hetero_graph_dqn":
                # Graph DQN系: 候補トラックIDリストを渡す
                # Last.fmの場合は track_metadata / グラフのトラックノード範囲に制限
                if USE_REAL_LASTFM_DATA:
                    n_graph_tracks = getattr(agent, "num_nodes", 256)
                    candidate_track_ids = list(range(n_graph_tracks))
                else:
                    candidate_track_ids = list(range(track_pool_size))
                action = agent.select_action(state, candidate_track_ids, epsilon)
                log_prob = None
            else:
                action = agent.select_action(state, cached_action_features, epsilon)
                log_prob = None
            
            next_state, reward, done = env.step(action)
            
            if AGENT_TYPE == "ppo":
                agent.store_transition(state, action, cached_action_features[action].cpu(), reward, next_state, done, log_prob)
            elif AGENT_TYPE == "graph_dqn" or AGENT_TYPE == "hetero_graph_dqn":
                # Graph DQN系: 独自のメモリ管理
                agent.store_transition(state, action, reward, next_state, done)
            else:
                agent.memory.push(state, action, cached_action_features[action].cpu(), reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        # エピソード終了後の学習
        # 報酬を記録して平均を計算（printより先に実行）
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-10:])
        
        if AGENT_TYPE == "ppo":
            actor_loss, critic_loss, entropy = agent.train_step(cached_action_features)
            if (episode + 1) % 10 == 0:
                print(f"Ep {episode+1:3d} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | A_Loss: {actor_loss:.4f} | C_Loss: {critic_loss:.4f}")
        elif AGENT_TYPE == "graph_dqn" or AGENT_TYPE == "hetero_graph_dqn":
            loss = agent.train_step()
            agent.update_target_network()
            agent.scheduler.step(avg_reward)
            if (episode + 1) % 10 == 0:
                tag = "Hetero" if AGENT_TYPE == "hetero_graph_dqn" else "Graph"
                print(f"Ep {episode+1:3d} | {tag} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | ε: {epsilon:.3f} | Loss: {loss:.4f}")
        else:
            loss = agent.train_step()
            agent.update_target_network()
            if (episode + 1) % 10 == 0:
                print(f"Ep {episode+1:3d} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | ε: {epsilon:.3f}")
        
        if AGENT_TYPE == "dqn":
            agent.scheduler.step(avg_reward)
        
        avg_resp = np.mean(env.response_history)
        
        writer.add_scalar('Episode/Total_Reward', total_reward, episode)
        writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Episode/Avg_Response', avg_resp, episode)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
    
    # training finished — keep writer open until after evaluation and copying
    

    # ======================================================
    #  テスト + モデル保存
    # ======================================================
    test_results = evaluate_agent(agent, env, cached_action_features, n_episodes=50)
    
    metrics = {
        'best_avg_reward': float(best_avg_reward),
        'test_results': test_results
    }
    
    # Graph DQN系（homogeneous / heterogeneous）は action_features 不要
    if AGENT_TYPE in ('graph_dqn', 'hetero_graph_dqn'):
        agent.save_model(save_dir, metrics)
    else:
        agent.save_model(save_dir, metrics, cached_action_features)
    # Log evaluation metrics to TensorBoard
    try:
        writer.add_scalar('Test/Mean_Reward', float(test_results['avg_reward']), 0)
        writer.add_scalar('Test/Std_Reward', float(test_results['std_reward']), 0)
        writer.add_scalar('Test/Mean_Response', float(test_results.get('avg_response', 0.0)), 0)
    except Exception:
        pass
    
    # 評価結果サマリーをTensorBoardに保存（Markdown形式）
    try:
        eval_summary = f"""### 評価結果

| 指標 | 値 |
|------|-----|
| **平均報酬** | {test_results['avg_reward']:.2f} ± {test_results['std_reward']:.2f} |
| **平均応答率** | {test_results.get('avg_response', 0.0):.3f} ± {test_results.get('std_response', 0.0):.3f} |
| **重複率** | {test_results.get('avg_duplicate_rate', 0.0):.1%} |
| **報酬範囲** | [{test_results.get('min_reward', 0):.2f}, {test_results.get('max_reward', 0):.2f}] |
"""
        writer.add_text('Evaluation/Summary', eval_summary, 0)
    except Exception as e:
        print(f"TensorBoard logging error: {e}")

    # Close writer after all logging done
    writer.close()

    # Copy tensorboard logs into the saved model directory so events are preserved
    try:
        dst_tb_dir = os.path.join(save_dir, 'tensorboard')
        shutil.copytree(log_dir, dst_tb_dir, dirs_exist_ok=True)
    except TypeError:
        # For older Python versions without dirs_exist_ok
        try:
            if os.path.exists(dst_tb_dir):
                shutil.rmtree(dst_tb_dir)
            shutil.copytree(log_dir, dst_tb_dir)
        except Exception:
            # Non-fatal: just continue
            pass

    print(f"\n完了！保存先: {save_dir}")


if __name__ == "__main__":
    main()
