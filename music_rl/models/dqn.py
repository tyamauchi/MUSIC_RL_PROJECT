import torch
import torch.nn as nn


class DuelingActionHeadDQN(nn.Module):
    def __init__(self, state_dim, action_feature_dim):
        super(DuelingActionHeadDQN, self).__init__()
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action_features):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_features = self.state_net(state)
        
        if action_features.dim() == 2:
            if action_features.size(0) == state.size(0):
                action_features = action_features.unsqueeze(1)
            else:
                action_features = action_features.unsqueeze(0)

        batch_size = state_features.size(0)
        if action_features.size(0) == 1 and batch_size > 1:
            action_features = action_features.expand(batch_size, -1, -1)
        
        action_features_processed = self.action_net(action_features.view(-1, action_features.size(-1)))
        action_features_processed = action_features_processed.view(batch_size, -1, 64)
        
        state_features_expanded = state_features.unsqueeze(1).expand(-1, action_features.size(1), -1)
        
        value = self.value_stream(state_features)
        num_actions = action_features.size(1)
        value = value.expand(-1, num_actions)
        
        combined = torch.cat([state_features_expanded, action_features_processed], dim=-1)
        advantage = self.advantage_stream(combined.view(-1, combined.size(-1))).view(batch_size, -1)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values