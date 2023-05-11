import torch
import torch.nn as nn

class GraphEmbedding(nn.Module):
    def __init__(self, num_e_labels, e_dim):
        super().__init__()
        self.node_embed = nn.Embedding.from_pretrained(torch.load("data/initial_node_embedding.pt"))
        self.edge_embed = nn.Embedding(num_e_labels, e_dim) # randomly initialized
    
    def forward(self, bg):
        """input: batched DGLGraphs"""
        bg.ndata["node_feat"] = self.node_embed(bg.ndata["node_ids"])
        bg.edata["edge_feat"] = self.edge_embed(bg.edata["edge_ids"])
        return bg