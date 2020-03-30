import torch
import numpy as np
import torch.nn as nn

# code adapted from https://github.com/tkipf/c-swm/blob/master/modules.py

class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_agents,
                output_dim=1,  num_time_steps=5):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.num_time_steps = num_time_steps
        self.output_dim = output_dim
        self.edge_list = None
        self.num_time_steps = 0

        self.node_mlp = nn.Sequential(
            nn.Linear(num_time_steps * input_dim *2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )    

    def _node_model(self, source, target):
        out = torch.cat([source, target], dim=0)
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, num_time_steps, num_agents):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.num_time_steps != num_time_steps:
            self.num_time_steps = num_time_steps

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_agents, num_agents)

            # Remove diagonal.
            adj_full -= torch.eye(num_agents)
            self.edge_list = adj_full.nonzero()

            # Copy `num_time_steps` times and add offset.
            self.edge_list = self.edge_list.repeat(num_time_steps, 1)
            offset = torch.arange(
                0, num_time_steps * num_agents, num_agents).unsqueeze(-1)
            offset = offset.expand(num_time_steps, num_agents * (num_agents - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

        return self.edge_list

    def forward(self, states, action):
       
        num_time_steps = states.size(0)
        num_nodes = states.size(1)

        # states: [num_time_steps (B), num_agents, input_dim]
        # input_attr: Flatten states tensor to [num_agents, B * input_dim, 1]
        input_attr = states.permute(1,0,2).view(states.size(0),-1,1)

        if num_nodes > 1:
            # edge_index: [B * (num_agents*[num_agents-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                num_time_steps, num_nodes)

            row, col = edge_index
            node_attr = self._node_model(
                input_attr[row], input_attr[col])
        return node_attr.view(num_time_steps, num_nodes, -1)

        