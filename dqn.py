import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np



class DQN(nn.Module):
    def __init__(self, num_agents, num_actions, hidden_dim, num_time_steps):
        super(DQN, self).__init__()
        state_dim = num_actions*num_time_steps
        self.num_agents = num_agents
        self.num_time_steps = num_time_steps
        self.num_actions = num_actions
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.ReLU()
        )

    def forward(self, input):
        # Input is a state representation with shape [batch_size=1,2*h]
        return self.q_net(input)

    def select_action(self, q_val, epsilon=0.05):
        # Apply epsilon-greedy action selection
        is_random = np.random.rand()
        if is_random > epsilon:
            return F.gumbel_softmax(logits=q_val, hard=True)
        else:
            act = F.one_hot(torch.arange(0, self.num_actions)).type(torch.FloatTensor)
            rand_idx = np.random.randint(0, self.num_actions)
            act = act[rand_idx,:].reshape(-1, self.num_actions)
            #print('explore actioin has shape:', act.shape)
            return act

     



