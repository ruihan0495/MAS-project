import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple
from itertools import count



class DQN(nn.Module):
    def __init__(self, env, num_agents, num_actions, hidden_dim, num_time_steps):
        super(DQN, self).__init__()
        state_dim = num_actions*num_time_steps
        self.num_agents = num_agents
        self.num_time_steps = num_time_steps
        self.env = env
        self.num_actions = num_actions
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.ReLU()
        )

    def forward(self, input):
        # Input is a state representation with shape [batch_size=1,2,h]
        return self.q_net(input)

    def select_action(self, state, epsilon=0.05):
        # Apply epsilon-greedy action selection
        is_random = np.random.rand()
        if is_random > epsilon:
            return self.q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)



