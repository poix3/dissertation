import torch
import torch.nn as nn
from transformers import BertModel
from layer.graphTransformer_layer import GraphTransformerLayer

class Encoder(nn.Module):
    def __init__(self, id = 0, in_dim=768, out_dim=768, num_heads=4):
        super().__init__()
        self.graph_encoder = GraphTransformerLayer(in_dim, out_dim, num_heads)
        self.cross_segment_encoder = self.apply_bert_weight(id)

    @torch.no_grad()
    def apply_bert_weight(self, id):
        bert = BertModel.from_pretrained("bert-base-uncased")
        return bert.encoder.layer[id]

    def forward(self, bg):
        """input: batched DGLGraphs"""
        bg = self.graph_encoder(bg)
        mask = bg.ndata["top"]
        top_feat = bg.ndata["node_feat"][mask == 1]
        top_feat = torch.unsqueeze(top_feat, dim=0) # fit transformer
        top_feat = self.cross_segment_encoder(top_feat)[0]
        top_feat = torch.squeeze(top_feat)
        bg.ndata["node_feat"][mask == 1] = top_feat # update top features
        return bg, top_feat