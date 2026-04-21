"""
gnn/model.py
------------
2-layer Graph Attention Network (GAT) for node classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AMLGNN(nn.Module):
    def __init__(self, in_dim: int = 10):
        super().__init__()
        # Layer 1: GAT with 4 heads, concat=True (output dim = 32 * 4 = 128)
        self.gat1 = GATConv(in_dim, 32, heads=4, dropout=0.3)
        
        # Layer 2: GAT with 1 head, concat=False (output dim = 32)
        self.gat2 = GATConv(32 * 4, 32, heads=1, concat=False, dropout=0.3)
        
        # Final projection to logit
        self.lin = nn.Linear(32, 1)

    def forward(self, x, edge_index):
        # x: [Nodes, in_dim]
        # edge_index: [2, Edges]
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        
        # Output: [Nodes, 1] -> [Nodes]
        x = self.lin(x)
        return x.squeeze()
