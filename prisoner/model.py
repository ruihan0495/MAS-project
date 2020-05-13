import torch
import torch.nn as nn
'''idea: use gnn to model partner selection phase,
it should try to predict the edge type. In general,
there are 2 edge types, i.e. selected and not_selected,
the output is also a graph with only one edge has type
selected. The game playing phase is modeled with Q-learning.'''

class GNNSelect(nn.Module):
    def __init__(self, total_agent, past_steps, hidden_dim, batch_size):        
        super(GNNSelect, self).__init__()
        self.num_agents = total_agent
        self.edge_types = [0,1] # 0 not selected, 1 selected
        self.edge_list = None
        input_dim = 2 * past_steps * 2 + 2
        self.num_batches = batch_size
        #self.node_mlp = nn.Sequential()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 2)
        )

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

    def get_rel(self, row, col, rel):
        # Return a 2-dimentional one-hot vector
        type_id = rel[row,col].long()
        print(type_id)
        return nn.functional.one_hot(type_id, num_classes=2)

    def forward(self, rel, input):
        # Input has shape [num_agents-1, 2, h] assume batchsize=1
        edge_index = self._get_edge_list_fully_connected(1,self.num_agents)
        row, col = edge_index
        sender = input[row,:,:]
        receiver = input[col,:,:]
        etype = self.get_rel(row, col, rel).float().unsqueeze(-1)
        edge = torch.cat((sender, receiver, etype),1)
        rel = self.edge_mlp(edge.squeeze())
        return rel


# Test sample usage
def main():
    model = GNNSelect(4,1,6,1)
    input = torch.tensor([[0.,1.],[1.,0.],[1.,0.],[0.,1.]]).unsqueeze(-1)
    rel = torch.empty(4, 4).random_(2)
    rel = model.forward(rel,input)
    print(rel)
   
if __name__ == "__main__":
    main()          

