#!/usr/bin/env python3
"""
クロスバリデーションスクリプト
213059モデルと213819モデルを異なる環境でテストして真のパフォーマンスを評価
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, '/Users/yamauchitakuya/MyResearch/music_rl_project_experiment/music_rl')

from config.config import *
from simulators.user_simulator import LSTMUserSimulator
from agents.dqn_agent import DoubleDQNAgent
from environment.music_env import MusicEnvironment
from utils.evaluation import evaluate_agent


def load_model_and_simulator(model_dir):
    """モデルとシミュレータを読み込む"""
    # LSTMシミュレータ
    lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
    lstm_path = os.path.join(model_dir, 'user_simulator.pth')
    lstm_sim.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    lstm_sim.eval()
    
    # RLエージェント
    agent = DoubleDQNAgent(STATE_DIM, ACTION_FEATURE_DIM, use_dueling=True)
    
    policy_path = os.path.join(model_dir, 'policy_net.pth')
    agent.policy_net.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    agent.policy_net.eval()
    
    # Action features
    action_features_path = os.path.join(model_dir, 'action_features.pth')
    action_features = torch.load(action_features_path, map_location=DEVICE)
    
    return agent, lstm_sim, action_features


def evaluate_cross_validation(agent, simulator, action_features, n_episodes=50):
    """クロスバリデーション評価"""
    env = MusicEnvironment(simulator, TRACK_POOL_SIZE, SESSION_LENGTH, STATE_DIM)
    
    results = {
        'rewards': [],
        'responses': [],
        'session_details': []
    }
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        responses = []
        
        for step in range(SESSION_LENGTH):
            # Greedyアクション（epsilon=0）
            action = agent.select_action(state, action_features, epsilon=0.0)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            responses.append(env.response_history[-1] if env.response_history else 0)
            state = next_state
        
        results['rewards'].append(total_reward)
        results['responses'].append(np.mean(responses))
        results['session_details'].append({
            'reward': total_reward,
            'avg_response': np.mean(responses),
            'unique_tracks': len(set(env.track_history))
        })
    
    return {
        'mean_reward': float(np.mean(results['rewards'])),
        'std_reward': float(np.std(results['rewards'])),
        'min_reward': float(np.min(results['rewards'])),
        'max_reward': float(np.max(results['rewards'])),
        'mean_response': float(np.mean(results['responses'])),
        'std_response': float(np.std(results['responses'])),
        'session_details': results['session_details']
    }


def main():
    print("=" * 80)
    print("クロスバリデーション: 213059 vs 213819")
    print("=" * 80)
    print()
    
    # モデルディレクトリ
    model_213059_dir = '/Users/yamauchitakuya/MyResearch/music_rl_project_experiment/music_rl/saved_models/20260216_213059'
    model_213819_dir = '/Users/yamauchitakuya/MyResearch/music_rl_project_experiment/music_rl/saved_models/20260216_213819'
    
    # モデルとシミュレータの読み込み
    print("モデルを読み込み中...")
    agent_213059, sim_213059, features_213059 = load_model_and_simulator(model_213059_dir)
    agent_213819, sim_213819, features_213819 = load_model_and_simulator(model_213819_dir)
    print("✅ モデル読み込み完了\n")
    
    # 4つのテストケース
    test_cases = [
        {
            'name': '213059 on 213059 (オリジナル)',
            'agent': agent_213059,
            'simulator': sim_213059,
            'features': features_213059,
            'description': '213059モデルを自身のシミュレータでテスト（ベースライン）'
        },
        {
            'name': '213059 on 213819 (クロステスト)',
            'agent': agent_213059,
            'simulator': sim_213819,
            'features': features_213059,
            'description': '213059モデルを213819のシミュレータでテスト'
        },
        {
            'name': '213819 on 213819 (オリジナル)',
            'agent': agent_213819,
            'simulator': sim_213819,
            'features': features_213819,
            'description': '213819モデルを自身のシミュレータでテスト（ベースライン）'
        },
        {
            'name': '213819 on 213059 (クロステスト)',
            'agent': agent_213819,
            'simulator': sim_213059,
            'features': features_213819,
            'description': '213819モデルを213059のシミュレータでテスト'
        }
    ]
    
    all_results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/4] {test_case['name']}")
        print(f"      {test_case['description']}")
        print("-" * 80)
        
        results = evaluate_cross_validation(
            test_case['agent'],
            test_case['simulator'],
            test_case['features'],
            n_episodes=50
        )
        
        all_results[test_case['name']] = results
        
        print(f"  平均報酬:   {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"  報酬範囲:   [{results['min_reward']:.4f}, {results['max_reward']:.4f}]")
        print(f"  平均応答率: {results['mean_response']:.4f} ± {results['std_response']:.4f}")
        print()
    
    # 結果の比較分析
    print("\n" + "=" * 80)
    print("分析結果")
    print("=" * 80)
    
    # 213059の汎化能力
    r_213059_on_213059 = all_results['213059 on 213059 (オリジナル)']['mean_reward']
    r_213059_on_213819 = all_results['213059 on 213819 (クロステスト)']['mean_reward']
    degradation_213059 = ((r_213059_on_213059 - r_213059_on_213819) / r_213059_on_213059) * 100
    
    print(f"\n【213059の汎化能力】")
    print(f"  自環境:    {r_213059_on_213059:.4f}")
    print(f"  他環境:    {r_213059_on_213819:.4f}")
    print(f"  性能低下:  {degradation_213059:.2f}%")
    
    if abs(degradation_213059) < 5:
        print(f"  評価: ✅ 良好な汎化（5%以内の低下）")
    elif abs(degradation_213059) < 15:
        print(f"  評価: ⚠️  中程度の汎化（5-15%の低下）")
    else:
        print(f"  評価: ❌ 汎化能力に問題（15%以上の低下）")
    
    # 213819の汎化能力
    r_213819_on_213819 = all_results['213819 on 213819 (オリジナル)']['mean_reward']
    r_213819_on_213059 = all_results['213819 on 213059 (クロステスト)']['mean_reward']
    degradation_213819 = ((r_213819_on_213819 - r_213819_on_213059) / r_213819_on_213819) * 100
    
    print(f"\n【213819の汎化能力】")
    print(f"  自環境:    {r_213819_on_213819:.4f}")
    print(f"  他環境:    {r_213819_on_213059:.4f}")
    print(f"  性能低下:  {degradation_213819:.2f}%")
    
    if abs(degradation_213819) < 5:
        print(f"  評価: ✅ 良好な汎化（5%以内の低下）")
    elif abs(degradation_213819) < 15:
        print(f"  評価: ⚠️  中程度の汎化（5-15%の低下）")
    else:
        print(f"  評価: ❌ 汎化能力に問題（15%以上の低下）")
    
    # 総合比較
    print(f"\n【総合評価】")
    print(f"  213059 平均: {(r_213059_on_213059 + r_213059_on_213819) / 2:.4f}")
    print(f"  213819 平均: {(r_213819_on_213819 + r_213819_on_213059) / 2:.4f}")
    
    # 重要な発見
    print(f"\n【重要な発見】")
    
    # 213059環境での比較
    if r_213819_on_213059 > r_213059_on_213059:
        diff = r_213819_on_213059 - r_213059_on_213059
        print(f"  ✅ 213819は213059環境でも優位 (+{diff:.4f})")
        print(f"     → 213819は真に優れている可能性")
    else:
        diff = r_213059_on_213059 - r_213819_on_213059
        print(f"  ⚠️  213059は自環境で優位 (+{diff:.4f})")
        print(f"     → 各モデルは自環境に特化している")
    
    # 213819環境での比較
    if r_213059_on_213819 > r_213819_on_213819:
        diff = r_213059_on_213819 - r_213819_on_213819
        print(f"  ⚠️  213059は213819環境でも優位 (+{diff:.4f})")
        print(f"     → 予想外の結果、要調査")
    else:
        diff = r_213819_on_213819 - r_213059_on_213819
        print(f"  ✅ 213819は自環境で優位 (+{diff:.4f})")
        print(f"     → 期待通りの結果")
    
    # 過学習の兆候
    print(f"\n【過学習の兆候】")
    if degradation_213819 > degradation_213059 + 10:
        print(f"  🚨 213819は過学習の可能性が高い")
        print(f"     性能低下: 213819 ({degradation_213819:.1f}%) > 213059 ({degradation_213059:.1f}%)")
    elif degradation_213059 > degradation_213819 + 10:
        print(f"  🚨 213059は過学習の可能性が高い")
        print(f"     性能低下: 213059 ({degradation_213059:.1f}%) > 213819 ({degradation_213819:.1f}%)")
    else:
        print(f"  ✅ 両モデルとも同程度の汎化能力")
    
    # 結果をJSONで保存
    output_dir = '/Users/yamauchitakuya/MyResearch/music_rl_project_experiment/music_rl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'cross_validation_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 詳細結果を保存: {output_file}")
    print("\n" + "=" * 80)
    print("クロスバリデーション完了")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()