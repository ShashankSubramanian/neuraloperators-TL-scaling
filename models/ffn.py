import torch
import torch.nn as nn
import numpy as np
from utils.misc_utils import set_activation

class FeedForward(nn.Module):
    ''' An n-layer-feed-forward-layer module '''
    def __init__(self, in_dim=2, out_dim=1, depth=5, hidden_dim=50, activation='tanh'):
        super().__init__()
        self.depth = depth
        self.activation = set_activation(activation)
        self.ff_in = nn.Linear(in_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(self.depth-2)])
        self.ff_out = nn.Linear(hidden_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        ''' Xavier Normal Initialization '''
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.ff_in(x)
        x = self.activation(x)
        for i in range(self.depth-2):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.ff_out(x)
        return x

def ffn_pinns(params):
   return FeedForward(in_dim=params.in_dim, out_dim=params.out_dim,
                      depth=params.depth, 
                      hidden_dim=params.hidden_dim,
                      activation='tanh')
