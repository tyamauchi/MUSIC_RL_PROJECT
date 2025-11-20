import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from config.config import DEVICE, BATCH_SIZE, GAMMA, LEARNING_RATE, UPDATE_TARGET_FREQ
from models.dqn import DuelingActionHeadDQN


class DoubleDQNAgent:
    def __init__(self, state_dim, action_feature_dim, use_dueling=True):
        self.use_dueling = use_dueling
        
        if use_dueling:
            self.policy_net = DuelingActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
            self.target_net = DuelingActionHeadDQN(state_dim, action_feature_dim).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=30, min_lr=5e-5
        )
        
        self.memory = None
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state, action_features, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                q_values = self.policy_net(state_tensor, action_features)
                
                track_history = np.array(state)[:20]
                penalties = torch.zeros_like(q_values)
                for i in range(action_features.size(0)):
                    if i in track_history:
                        penalties[0, i] = -2.0
                
                return torch.argmax(q_values + penalties).item()
        else:
            return random.randrange(action_features.size(0))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        states, actions, action_features, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = states.to(DEVICE)
        action_features = action_features.to(DEVICE)
        rewards = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)
        
        current_q = self.policy_net(states, action_features).squeeze(1)
        
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states, action_features)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_net(next_states, action_features)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        return loss.item()
    
    def update_target_network(self):
        self.episodes_done += 1
        if self.episodes_done % UPDATE_TARGET_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  → ターゲットネットワーク更新 (Episode {self.episodes_done})")
    
    def save_model(self, save_dir, metrics, action_features):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(save_dir, 'policy_net.pth'))
        torch.save(action_features, os.path.join(save_dir, 'action_features.pth'))
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())
