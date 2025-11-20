import numpy as np
import torch
import torch.nn as nn
from config.config import LSTM_HIDDEN_SIZE, DEVICE


class DeterministicUserSimulator:
    def __init__(self, track_pool_size=256):
        np.random.seed(42)
        self.track_preferences = np.random.rand(track_pool_size) * 0.4 + 0.6
        self.track_genres = np.random.randint(0, 10, track_pool_size)
        self.genre_preferences = None
        
    def reset_user_state(self):
        self.genre_preferences = np.random.rand(10) * 0.4 + 0.6
        
    def get_response(self, state, action, step):
        action_idx = int(action)
        base_response = self.track_preferences[action_idx]
        genre_match = self.genre_preferences[self.track_genres[action_idx]]
        base_response = base_response * 0.6 + genre_match * 0.4
        
        state_array = np.array(state)
        track_history = state_array[:len(state_array)//2]
        response_history = state_array[len(state_array)//2:]
        
        if step > 0:
            prev_responses = [r for r in response_history[:step] if r > 0]
            if len(prev_responses) > 0:
                recent_avg = np.mean(prev_responses[-3:])
                if abs(base_response - recent_avg) < 0.15:
                    base_response += 0.05
        
        if action_idx in track_history[:step]:
            base_response *= 0.3
        
        base_response += np.random.normal(0, 0.02)
        return np.clip(base_response, 0.0, 1.0)


class LSTMUserSimulator(nn.Module):
    def __init__(self, input_size, hidden_sizes=LSTM_HIDDEN_SIZE):
        super(LSTMUserSimulator, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[2], 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        out1, hidden1 = self.lstm1(x, hidden[0] if hidden else None)
        out2, hidden2 = self.lstm2(out1, hidden[1] if hidden else None)
        out3, hidden3 = self.lstm3(out2, hidden[2] if hidden else None)
        
        completion_prob = self.output_layer(out3)
        return completion_prob, (hidden1, hidden2, hidden3)