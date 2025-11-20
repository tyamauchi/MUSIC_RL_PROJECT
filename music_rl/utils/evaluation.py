import numpy as np


def evaluate_agent(agent, env, cached_action_features, n_episodes=50):
    print("\n" + "=" * 60)
    print("エージェント評価開始（テストセット）")
    print("=" * 60)
    
    total_rewards = []
    avg_responses = []
    duplicate_rates = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.session_length):
            action = agent.select_action(state, cached_action_features, epsilon=0.0)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        avg_responses.append(np.mean(env.response_history))
        
        unique_tracks = len(set([t for t in env.track_history if t > 0]))
        duplicate_rate = 1.0 - (unique_tracks / env.session_length)
        duplicate_rates.append(duplicate_rate)
    
    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_response': np.mean(avg_responses),
        'std_response': np.std(avg_responses),
        'avg_duplicate_rate': np.mean(duplicate_rates),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }
    
    print(f"\n【評価結果】({n_episodes}エピソード)")
    print(f"  平均報酬: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均応答率: {results['avg_response']:.3f} ± {results['std_response']:.3f}")
    print(f"  重複率: {results['avg_duplicate_rate']:.1%}")
    print(f"  報酬範囲: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("=" * 60 + "\n")
    
    return results