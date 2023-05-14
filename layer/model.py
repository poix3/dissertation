import dgl
import torch
import torch.nn as nn

from layer.graphEmbedding_layer import GraphEmbedding
from layer.encoder_block import Encoder

class Net(nn.Module):
    def __init__(self, num_layer, num_e_labels, e_dim = 768, out_dim = 768):
        super().__init__()
        self.embedding = GraphEmbedding(num_e_labels, e_dim)
        self.encoder = nn.ModuleList([Encoder(id) for id in range(num_layer)])
        self.max_pooling_proj = nn.Sequential(nn.Linear(out_dim, out_dim//6),
                                   nn.Tanh(),
                                   nn.Linear(out_dim//6, 10))

    def forward(self, gs):
        # TODO
        """input: a list DGLGraphs"""
        bg = dgl.batch(gs)
        bg = self.embedding(bg)
        for layer in self.encoder:
            bg, cse_out = layer(bg)
        
        pooled = torch.max(cse_out, dim=0).values
        logits = self.max_pooling_proj(pooled)
        return logits