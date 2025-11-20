import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def generate_expert_trajectories(deterministic_sim, n_trajectories=2000, session_length=20, add_exploration=True):
    print("=" * 60)
    print("高品質な学習データを生成中...")
    print("=" * 60)
    
    trajectories = []
    track_pool_size = 256
    
    for traj_idx in range(n_trajectories):
        deterministic_sim.reset_user_state()
        track_history = []
        response_history = []
        exploration_rate = 0.1 if add_exploration else 0.0
        
        for step in range(session_length):
            current_state = track_history + [0.0] * (20 - len(track_history))
            current_state += response_history + [0.0] * (20 - len(response_history))
            
            if add_exploration and np.random.random() < exploration_rate:
                available_tracks = [t for t in range(track_pool_size) if t not in track_history]
                best_action = np.random.choice(available_tracks)
                best_response = deterministic_sim.get_response(current_state, best_action, step)
            else:
                candidates = np.random.choice(track_pool_size, size=50, replace=False)
                best_action = None
                best_response = -1
                
                for candidate in candidates:
                    if candidate not in track_history:
                        temp_response = deterministic_sim.get_response(current_state, candidate, step)
                        if temp_response > best_response:
                            best_response = temp_response
                            best_action = candidate
                
                if best_action is None:
                    best_action = np.random.randint(0, track_pool_size)
                    best_response = deterministic_sim.get_response(current_state, best_action, step)
            
            state_with_action = current_state + [float(best_action)]
            trajectories.append((state_with_action, best_response))
            
            track_history.append(best_action)
            response_history.append(best_response)
        
        if (traj_idx + 1) % 500 == 0:
            avg_quality = np.mean([t[1] for t in trajectories[-10000:]])
            print(f"軌跡生成 {traj_idx + 1}/{n_trajectories}, 平均応答率: {avg_quality:.3f}")
    
    print("=" * 60)
    print(f"合計 {len(trajectories)} ステップの学習データを生成")
    print("=" * 60)
    return trajectories


def pretrain_lstm_with_expert_data(lstm_sim, trajectories, n_epochs=20, batch_size=64):
    print("\nLSTMシミュレータの事前学習開始...")
    from config.config import DEVICE
    
    optimizer = optim.Adam(lstm_sim.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    for epoch in range(n_epochs):
        random.shuffle(trajectories)
        epoch_losses = []
        
        for i in range(0, len(trajectories) - batch_size, batch_size):
            batch = trajectories[i:i+batch_size]
            states = torch.FloatTensor([t[0] for t in batch]).unsqueeze(1).to(DEVICE)
            targets = torch.FloatTensor([t[1] for t in batch]).unsqueeze(1).unsqueeze(2).to(DEVICE)
            
            outputs, _ = lstm_sim(states)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_sim.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    print(f"事前学習完了! 最終Loss: {avg_loss:.4f}\n")