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
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.Tanh(),
                                  nn.Linear(out_dim, 10))

    def get_attention_mask(self, bgs):
        # record number of top nodes
        top_count = []
        for bg in bgs:
            top_mask = bg.ndata["top_mask"]
            num_top = sum(top_mask == 1)
            top_count.append(num_top)
        # create attention mask
        attention_mask = []
        max_len = max(top_count)
        for cnt in top_count:
            row = [1] * cnt + [0] * (max_len - cnt)
            attention_mask.append(row)
        
        attention_mask = torch.FloatTensor(attention_mask)
        return attention_mask
    
    def max_pooling(self, cse_out, attention_mask):
        """cse_out, dim: [batch_size, max_num_top_nodes, feat_dim]
           attention_mask, dim: [batch_size, max_num_top_nodes]"""
        pooled = []
        for out, att_mask in zip(cse_out, attention_mask):
            num_top = sum(att_mask == 1)
            out = out[:num_top]
            pooled.append(torch.max(out, dim=0).values)

        return torch.stack(pooled, dim=0)

    def forward(self, bgs):
        """input: a list batched DGLGraphs"""
        # record information for unbatch operation
        node_split = torch.tensor([bg.num_nodes() for bg in bgs])
        edge_split = torch.tensor([bg.num_edges() for bg in bgs])

        attention_mask = self.get_attention_mask(bgs) # dim: [batch_size, max_num_top_nodes]
        bgs = [self.embedding(bg) for bg in bgs]
        bbg = dgl.batch(bgs)
        for layer in self.encoder:
            bbg, cse_out = layer(bbg, attention_mask, node_split, edge_split)
        
        pooled = self.max_pooling(cse_out, attention_mask) # dim: [batch_size, feat_dim]
        logits = self.proj(pooled)
        return logits