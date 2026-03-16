import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "training_data.csv")
MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_model.pth")

sys.path.append(SCRIPT_DIR)
from environment import NetworkEnv5G

class DQNNetwork(nn.Module):
    def __init__(self, state_size=6, action_size=2):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.linear(state_size, 64)
            # Linear layer: 6 inputs → 64 outputs
            # Each of 64 neurons sees all 6 features
            # Learns weights: output = weight * input + bias

            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.apend((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        """Returns how many experiences are stored."""
        return len(self.buffer)