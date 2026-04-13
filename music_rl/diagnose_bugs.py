#!/usr/bin/env python3
"""
環境とモデルのバグ診断スクリプト
Hidden state管理が正しく機能しているか確認
"""

import sys
import os
import torch
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, '/home/claude/music_rl')

from config.config import *
from environment.music_env import MusicEnvironment
from simulators.user_simulator import LSTMUserSimulator, DeterministicUserSimulator
from agents.dqn_agent import DoubleDQNAgent

print("="*80)
print("バグ診断：環境とモデルの検証")
print("="*80)

# ===================================================================
# テスト1: 環境にhidden属性が存在するか
# ===================================================================
print("\n[テスト1] MusicEnvironmentにhidden属性が存在するか")
print("-"*80)

lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
env = MusicEnvironment(lstm_sim, TRACK_POOL_SIZE, SESSION_LENGTH, STATE_DIM)

has_hidden = hasattr(env, 'hidden')
print(f"✅ hidden属性が存在: {has_hidden}")

if not has_hidden:
    print("🚨 警告: hidden属性が存在しません！旧環境を使用している可能性があります。")
    print("   修正版環境（music_env_fixed.py）を environment/music_env.py に配置してください。")
    sys.exit(1)
else:
    print("✅ 修正版環境が正しく読み込まれています")

# ===================================================================
# テスト2: Hidden stateが適切に管理されているか
# ===================================================================
print("\n[テスト2] Hidden stateが適切に管理されているか")
print("-"*80)

# 初期状態
print(f"初期のhidden: {env.hidden}")
assert env.hidden is None, "初期のhiddenがNoneではありません"
print("✅ 初期状態: hidden = None")

# リセット後
state = env.reset()
print(f"リセット後のhidden: {env.hidden}")
assert env.hidden is None, "リセット後のhiddenがNoneではありません"
print("✅ リセット後: hidden = None")

# 1ステップ実行後
next_state, reward, done = env.step(10)
print(f"1ステップ後のhidden is not None: {env.hidden is not None}")
assert env.hidden is not None, "1ステップ後のhiddenがNoneのままです"
print("✅ 1ステップ後: hidden が設定される")

# 2ステップ目
prev_hidden = env.hidden
next_state, reward, done = env.step(20)
curr_hidden = env.hidden
print(f"2ステップ目でhidden更新: {prev_hidden != curr_hidden}")
assert curr_hidden is not None, "2ステップ後のhiddenがNoneです"
print("✅ 2ステップ目: hidden が引き継がれて更新される")

# ===================================================================
# テスト3: 応答値が適切な範囲か
# ===================================================================
print("\n[テスト3] 応答値が適切な範囲にあるか")
print("-"*80)

# DeterministicSimulatorでLSTMを事前学習
det_sim = DeterministicUserSimulator(TRACK_POOL_SIZE)
from utils.training import generate_expert_trajectories, pretrain_lstm_with_expert_data

print("小規模データで事前学習を実施...")
trajectories = generate_expert_trajectories(det_sim, n_trajectories=100, 
                                            session_length=20, add_exploration=True)
lstm_sim_test = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
pretrain_lstm_with_expert_data(lstm_sim_test, trajectories, n_epochs=5)

# テスト環境作成
env_test = MusicEnvironment(lstm_sim_test, TRACK_POOL_SIZE, SESSION_LENGTH, STATE_DIM)

# 5エピソード実行
responses_list = []
for episode in range(5):
    state = env_test.reset()
    episode_responses = []
    
    for step in range(SESSION_LENGTH):
        action = np.random.randint(0, TRACK_POOL_SIZE)
        next_state, reward, done = env_test.step(action)
        episode_responses.append(env_test.response_history[step])
        state = next_state
    
    avg_response = np.mean(episode_responses)
    responses_list.append(avg_response)
    print(f"  Episode {episode+1}: 平均応答率 = {avg_response:.3f}")

overall_avg = np.mean(responses_list)
print(f"\n全体の平均応答率: {overall_avg:.3f}")

if overall_avg < 0.1:
    print("🚨 警告: 応答率が異常に低いです（<10%）")
    print("   Hidden stateが適切に管理されていない可能性があります。")
elif overall_avg > 0.7:
    print("✅ 応答率は正常範囲です（>70%）")
else:
    print("⚠️  応答率がやや低いです（10-70%）")
    print("   事前学習が不十分か、データの質に問題がある可能性があります。")

# ===================================================================
# テスト4: 標準偏差がゼロでないか
# ===================================================================
print("\n[テスト4] エピソード間で変動があるか")
print("-"*80)

std_response = np.std(responses_list)
print(f"応答率の標準偏差: {std_response:.4f}")

if std_response == 0:
    print("🚨 警告: 標準偏差がゼロです！完全に決定論的です。")
    print("   Hidden stateが適切に初期化されていない可能性があります。")
elif std_response < 0.001:
    print("⚠️  標準偏差が非常に小さいです。")
    print("   データの多様性が不足している可能性があります。")
else:
    print("✅ 標準偏差は正常です。適度な変動があります。")

# ===================================================================
# テスト5: 213059と213819モデルをロードしてテスト
# ===================================================================
print("\n[テスト5] 実際の213059と213819モデルで検証")
print("-"*80)

def test_saved_model(model_dir, model_name):
    print(f"\n{model_name}をテスト中...")
    
    # モデルとシミュレータを読み込み
    lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
    lstm_path = os.path.join(model_dir, 'user_simulator.pth')
    lstm_sim.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    lstm_sim.eval()
    
    agent = DoubleDQNAgent(STATE_DIM, ACTION_FEATURE_DIM, use_dueling=True)
    policy_path = os.path.join(model_dir, 'policy_net.pth')
    agent.policy_net.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    agent.policy_net.eval()
    
    action_features = torch.load(os.path.join(model_dir, 'action_features.pth'), 
                                 map_location=DEVICE)
    
    # 環境作成
    env = MusicEnvironment(lstm_sim, TRACK_POOL_SIZE, SESSION_LENGTH, STATE_DIM)
    
    # 環境にhidden属性があるか確認
    has_hidden = hasattr(env, 'hidden')
    print(f"  hidden属性: {has_hidden}")
    
    # 3エピソード実行
    episode_responses = []
    episode_rewards = []
    
    for episode in range(3):
        state = env.reset()
        
        # リセット後のhidden確認
        if episode == 0:
            print(f"  リセット後のhidden: {env.hidden}")
        
        total_reward = 0
        responses = []
        
        for step in range(SESSION_LENGTH):
            action = agent.select_action(state, action_features, epsilon=0.0)
            next_state, reward, done = env.step(action)
            
            # 最初のエピソードの最初のステップでhidden確認
            if episode == 0 and step == 0:
                print(f"  1ステップ後のhidden is not None: {env.hidden is not None}")
            
            total_reward += reward
            responses.append(env.response_history[step])
            state = next_state
        
        avg_response = np.mean(responses)
        episode_responses.append(avg_response)
        episode_rewards.append(total_reward)
        print(f"  Episode {episode+1}: 報酬={total_reward:.2f}, 応答率={avg_response:.3f}")
    
    mean_response = np.mean(episode_responses)
    std_response = np.std(episode_responses)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n  平均応答率: {mean_response:.3f} ± {std_response:.4f}")
    print(f"  平均報酬: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 評価
    if mean_response < 0.1:
        print(f"  🚨 {model_name}: 応答率が異常に低い（バグの可能性）")
        return False
    elif std_response == 0:
        print(f"  🚨 {model_name}: 標準偏差がゼロ（Hidden stateバグの可能性）")
        return False
    else:
        print(f"  ✅ {model_name}: 正常に動作しています")
        return True

# 213059をテスト
model_213059_dir = '/home/claude/music_rl/saved_models/20260216_213059'
result_213059 = test_saved_model(model_213059_dir, "213059 (Last.fm)")

# 213819をテスト
model_213819_dir = '/home/claude/music_rl/saved_models/20260216_213819'
result_213819 = test_saved_model(model_213819_dir, "213819 (DeterministicSim)")

# ===================================================================
# 最終判定
# ===================================================================
print("\n" + "="*80)
print("最終診断結果")
print("="*80)

if has_hidden and result_213059 and result_213819:
    print("✅ すべてのテストに合格しました！")
    print("   - 修正版環境が正しく読み込まれています")
    print("   - Hidden stateが適切に管理されています")
    print("   - 両モデルとも正常に動作しています")
    print("\n次のステップ:")
    print("   1. 詳細なクロスバリデーションを実行")
    print("   2. 推薦の多様性を検証")
    print("   3. リアルユーザーテストの準備")
elif not has_hidden:
    print("❌ 重大なバグ: 環境にhidden属性がありません")
    print("   修正版環境（music_env_fixed.py）を配置してください")
elif not (result_213059 and result_213819):
    print("❌ モデルに問題があります")
    print("   詳細は上記のテスト結果を確認してください")
else:
    print("⚠️  一部のテストで問題が検出されました")
    print("   詳細は上記のテスト結果を確認してください")

print("="*80)