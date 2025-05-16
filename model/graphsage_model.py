import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class GraphSAGE(nn.Module):
    def __init__(self, drug_input_dim, protein_input_dim, hidden_dim):
        super(GraphSAGE, self).__init__()

        # Initial linear layers to unify feature dims
        self.drug_lin = Linear(drug_input_dim, hidden_dim)
        self.protein_lin = Linear(protein_input_dim, hidden_dim)

        # HeteroConv layer
        self.conv1 = HeteroConv({
            ('drug', 'interacts', 'protein'): SAGEConv(hidden_dim, hidden_dim),
            ('protein', 'rev_interacts', 'drug'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='sum')

        # MLP for affinity prediction
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, drug_idx, protein_idx):
        # Project features into same space
        x_dict = {
            'drug': self.drug_lin(data['drug'].x),
            'protein': self.protein_lin(data['protein'].x)
        }

        # Message passing
        x_dict = self.conv1(x_dict, data.edge_index_dict)

        # Get embeddings for specific drug/protein pairs
        drug_embed = x_dict['drug'][drug_idx]
        protein_embed = x_dict['protein'][protein_idx]

        # Concatenate and predict
        combined = torch.cat([drug_embed, protein_embed], dim=1)
        out = self.fc(combined).squeeze(1)
        return out
