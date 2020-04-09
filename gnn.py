import torch
import numpy as np
import torch.nn as nn
import utils

# code adapted from https://github.com/tkipf/c-swm/blob/master/modules.py

class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_agents,
                output_dim,  num_time_steps=4):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.num_time_steps = num_time_steps
        self.output_dim = output_dim
        self.edge_list = None
       

        self.node_mlp = nn.Sequential(
            nn.Linear(num_time_steps * input_dim *2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )    

    def _node_model(self, source, target):
        out = torch.cat((source, target), dim=1)
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, num_batches, num_agents):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.num_batches != num_batches:
            self.num_batches = num_batches

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_agents, num_agents)

            # Remove diagonal.
            adj_full -= torch.eye(num_agents)
            self.edge_list = adj_full.nonzero()

            # Copy `num_batches` times and add offset.
            self.edge_list = self.edge_list.repeat(num_batches, 1)
            offset = torch.arange(
                0, num_batches * num_agents, num_agents).unsqueeze(-1)
            offset = offset.expand(num_batches, num_agents * (num_agents - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

        return self.edge_list

    def forward(self, states):
       
        num_batches = states.shape[0]
        num_nodes = states.shape[1]

        # states: [num_batches (B), num_agents, num_time_steps*input_dim]

        if num_nodes > 1:
            # edge_index: [2, B * (num_agents*[num_agents-1])] edge list
            edge_index = self._get_edge_list_fully_connected(
                num_batches, num_nodes)

            row, col = edge_index
            sender = torch.tensor(states[:,row,:].reshape((-1, self.num_time_steps*self.input_dim)))
            receiver = torch.tensor(states[:,col,:].reshape((-1, self.num_time_steps*self.input_dim)))
            node_attr = self._node_model(
                sender, receiver)
     
        # also it's better to convert the states into [batch_size, num_agents, time_step*out_dim]        
        return node_attr.view(num_batches, num_nodes, -1)

        
class PolicyNet():
    "policy net needs the input from gnn--predicted actions and input states information"
    def __init__(self, state_dim, hidden_dim, num_agents, time_step, output_dim=1):
        super(PolicyNet, self).__init__()
        input_dim = state_dim * time_step * num_agents
        self.q_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(), 
                                   nn.Linear(hidden_dim, output_dim) )

    def forward(self, states):
        # states have shape [batch_size, input_dim]
        out = self.q_net(states)
        out = utils.to_zero_one(out.numpy())  
        return out

