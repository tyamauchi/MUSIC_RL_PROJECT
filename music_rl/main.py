import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config.config import *
from simulators.user_simulator import DeterministicUserSimulator, LSTMUserSimulator
from agents.dqn_agent import DoubleDQNAgent
from environment.music_env import MusicEnvironment
from memory.replay_buffer import ReplayBuffer
from utils.training import generate_expert_trajectories, pretrain_lstm_with_expert_data
from utils.evaluation import evaluate_agent


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/music_rl_{timestamp}'
    save_dir = f'saved_models/{timestamp}'
    writer = SummaryWriter(log_dir)
    
    print("\n" + "=" * 60)
    print("Double DQN + Dueling DQN 音楽推薦システム")
    print("=" * 60 + "\n")
    
    det_sim = DeterministicUserSimulator(TRACK_POOL_SIZE)
    trajectories = generate_expert_trajectories(det_sim, n_trajectories=N_TRAJECTORIES, 
                                                session_length=SESSION_LENGTH, add_exploration=True)
    
    lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
    pretrain_lstm_with_expert_data(lstm_sim, trajectories, n_epochs=PRETRAIN_EPOCHS)
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(lstm_sim.state_dict(), os.path.join(save_dir, 'user_simulator.pth'))
    
    env = MusicEnvironment(lstm_sim, TRACK_POOL_SIZE, SESSION_LENGTH, STATE_DIM)
    agent = DoubleDQNAgent(STATE_DIM, ACTION_FEATURE_DIM, use_dueling=True)
    agent.memory = ReplayBuffer()
    
    best_avg_reward = float('-inf')
    rewards_history = []
    
    torch.manual_seed(RANDOM_SEED)
    cached_action_features = torch.randn(TRACK_POOL_SIZE, ACTION_FEATURE_DIM, device=DEVICE)
    
    print("学習開始\n")
    
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
        
        for step in range(SESSION_LENGTH):
            action = agent.select_action(state, cached_action_features, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, cached_action_features[action].cpu(), reward, next_state, done)
            
            loss = agent.train_step()
            total_reward += reward
            state = next_state
        
        agent.update_target_network()
        
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-10:])
        agent.scheduler.step(avg_reward)
        
        avg_resp = np.mean(env.response_history)
        
        writer.add_scalar('Episode/Total_Reward', total_reward, episode)
        writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Episode/Avg_Response', avg_resp, episode)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:3d} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | ε: {epsilon:.3f}")
    
    writer.close()
    
    test_results = evaluate_agent(agent, env, cached_action_features, n_episodes=50)
    
    metrics = {
        'best_avg_reward': float(best_avg_reward),
        'test_results': test_results
    }
    
    agent.save_model(save_dir, metrics, cached_action_features)
    print(f"\n完了！保存先: {save_dir}")


if __name__ == "__main__":
    main()