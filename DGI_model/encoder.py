import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LayerNorm
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, GravNetConv, TAGConv, ARMAConv, NNConv
from torch_geometric.nn import GraphNorm, GlobalAttention, Set2Set, DeepSetsAggregation, SetTransformerAggregation
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import DeepGraphInfomax
import random

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, conv, n_output, conv_params={}):
        super(Encoder, self).__init__()
        self.graph_norm0 = GraphNorm(4)        
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        self.act1 = nn.PReLU(hidden_channels)
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params)
        self.act2 = nn.PReLU(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.conv3 = conv(
            hidden_channels, hidden_channels, **conv_params)
        self.graph_norm3 = GraphNorm(hidden_channels)
        self.aggr = Set2Set(hidden_channels, 3)

    def forward(self, x, edge_index, batch=None):
        x = self.graph_norm0(x)
        x = self.act1(self.conv1(x, edge_index))
        x = self.graph_norm1(x)  
        
        x = self.act2(self.conv2(x, edge_index))
        x = self.graph_norm2(x)
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        
        batch = torch.zeros(x.shape[0],dtype=int).to(x.device)
        x = self.aggr(x, batch)     
        
        return x
