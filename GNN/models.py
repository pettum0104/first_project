import pandas as pd
import numpy as np
import os
import networkx as nx
import scipy
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LayerNorm
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, GravNetConv, TAGConv, ARMAConv, NNConv
from torch_geometric.nn import GraphNorm, GlobalAttention, Set2Set, DeepSetsAggregation, MLPAggregation, LSTMAggregation
import torch.nn as nn
import torch.optim as optim

class GNN3(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, conv, n_predict, conv_params={}):
        super(GNN3, self).__init__()
        
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params)
        self.conv3 = conv(
            hidden_channels, hidden_channels, **conv_params)
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, n_predict)
        self.graph_norm0 = GraphNorm(4)
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.graph_norm3 = GraphNorm(hidden_channels)
        nn_model = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, n_predict), 
        )
        self.aggr = GlobalAttention(nn_model)

    def forward(self, x, edge_index, batch):
        x = self.graph_norm0(x)
        x = self.conv1(x, edge_index)
        x = F.relu(self.graph_norm1(x))
        
#         x = F.dropout(x, p=0.5, training=self.training)    
        
        x = self.conv2(x, edge_index)
        x = F.relu(self.graph_norm2(x))
        
#         x = F.dropout(x, p=0.5, training=self.training)    
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        
        x = self.aggr(x, batch)
        
#         x = F.dropout(x, p=0.5, training=self.training)        
        x = F.relu(self.lin1(x))
        
#         x = F.dropout(x, p=0.5, training=self.training)    
        
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        
        return x
    
class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, conv, n_predict, conv_params={}):
        super(GNN, self).__init__()
        
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params)
        
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, n_predict)
        self.graph_norm0 = GraphNorm(4)
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.aggr = Set2Set(hidden_channels, 2)
        
    def forward(self, x, edge_index, batch):
        x = self.graph_norm0(x)
        x = self.conv1(x, edge_index)
        x = F.relu(self.graph_norm1(x, batch))
        
#         x = F.dropout(x, p=0.5, training=self.training)     
        
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x, batch)
        
        x = self.aggr(x, batch)
        
#         x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        
#         x = F.dropout(x, p=0.5, training=self.training)     
        
        x = self.lin2(x)
        
        return x
    