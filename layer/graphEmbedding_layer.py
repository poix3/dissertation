import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEmbedding(nn.Module):
    def __init__(self, num_e_labels, e_dim, hidden_size=768, hidden_dropout_prob=0.1):
        super().__init__()
        self.node_embed = nn.Embedding.from_pretrained(torch.load("data/initial_node_embedding.pt"))
        self.edge_embed = nn.Embedding(num_e_labels, e_dim) # randomly initialized
        self.pos_embed = nn.Embedding.from_pretrained(torch.load("data/initial_pos_embedding.pt"))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, bg):
        # NODE
        node_feat = self.node_embed(bg.ndata["node_ids"])
        top_mask = bg.ndata["top_mask"]
        top_feat = node_feat[top_mask == 1] # dim: [num_top_nodes, feat_dim]
        position_ids = torch.arange(top_feat.shape[0], device=DEVICE)
        # add position embedding to top nodes
        top_feat = top_feat + self.pos_embed(position_ids)
        node_feat[top_mask == 1] = top_feat
        bg.ndata["node_feat"] = self.dropout(node_feat)
        # EDGE
        edge_feat = self.edge_embed(bg.edata["edge_ids"])
        # layer norm
        edge_feat = self.layer_norm(edge_feat)
        bg.edata["edge_feat"] = self.dropout(edge_feat)
        return bg