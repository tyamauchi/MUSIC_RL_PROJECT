import random
import torch
from collections import deque, namedtuple
from config.config import MEMORY_SIZE


Experience = namedtuple('Experience', ['state', 'action', 'action_features', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, action_features, reward, next_state, done):
        self.buffer.append(Experience(state, action, action_features, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        action_features = torch.stack([e.action_features for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        return states, actions, action_features, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)