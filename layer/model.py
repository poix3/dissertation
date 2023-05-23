import dgl
import torch
import torch.nn as nn

from layer.graphEmbedding_layer import GraphEmbedding
from layer.graphTransformer_layer import GraphTransformerLayer
from layer.crossSegmentEncoder_layer import CrossSegmentEncoderLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, layout, num_e_labels=36, e_dim=768, out_dim=768):
        super().__init__()

        self.layout = layout
        self.embedding = GraphEmbedding(num_e_labels, e_dim)
        self.encoder = self.get_encoder()
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.Tanh(),
                                  nn.Linear(out_dim, 10))
    
    def get_encoder(self):
        if self.layout == "i":
            return Interleave()
        elif self.layout == "ec":
            return EarlyContextualization()
        elif self.layout == "lc":
            return LateContextualization()
        elif self.layout == "ah":
            return AH()
        else:
            raise ValueError("Invalid layout option.")

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
        
        attention_mask = torch.tensor(attention_mask, 
                                      dtype=torch.float,
                                      device=DEVICE)
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
        node_split = torch.tensor([bg.num_nodes() for bg in bgs], device=DEVICE)
        edge_split = torch.tensor([bg.num_edges() for bg in bgs], device=DEVICE)
        
        attention_mask = self.get_attention_mask(bgs) # dim: [batch_size, max_num_top_nodes]
        bgs = [self.embedding(bg.to(DEVICE)) for bg in bgs]
        bbg = dgl.batch(bgs)
        cse_out = self.encoder(bbg, attention_mask, node_split, edge_split)
        pooled = self.max_pooling(cse_out, attention_mask) # dim: [batch_size, feat_dim]
        logits = self.proj(pooled)
        return logits.to("cpu")

class Interleave(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layers = [2,5,8,11]
        self.graph_transformers = nn.ModuleList([GraphTransformerLayer() for _ in range(8)])
        self.cross_segment_encoders = nn.ModuleList([CrossSegmentEncoderLayer(id) for id in self.bert_layers])

    def forward(self, bbg, attention_mask, node_split, edge_split):
        bbg = self.graph_transformers[0](bbg)
        bbg = self.graph_transformers[1](bbg)
        bbg, cse_out = self.cross_segment_encoders[0](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[2](bbg)
        bbg = self.graph_transformers[3](bbg)
        bbg, cse_out = self.cross_segment_encoders[1](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[4](bbg)
        bbg = self.graph_transformers[5](bbg)
        bbg, cse_out = self.cross_segment_encoders[2](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[6](bbg)
        bbg = self.graph_transformers[7](bbg)
        bbg, cse_out = self.cross_segment_encoders[3](bbg, attention_mask, node_split, edge_split)
        return cse_out

class EarlyContextualization(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layers = [0,2,4,11]
        self.graph_transformers = nn.ModuleList([GraphTransformerLayer() for _ in range(8)])
        self.cross_segment_encoders = nn.ModuleList([CrossSegmentEncoderLayer(id) for id in self.bert_layers])
    
    def forward(self, bbg, attention_mask, node_split, edge_split):
        bbg, cse_out = self.cross_segment_encoders[0](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[0](bbg)
        bbg, cse_out = self.cross_segment_encoders[1](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[1](bbg)
        bbg, cse_out = self.cross_segment_encoders[2](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[2](bbg)
        bbg = self.graph_transformers[3](bbg)
        bbg = self.graph_transformers[4](bbg)
        bbg = self.graph_transformers[5](bbg)
        bbg = self.graph_transformers[6](bbg)
        bbg = self.graph_transformers[7](bbg)
        bbg, cse_out = self.cross_segment_encoders[3](bbg, attention_mask, node_split, edge_split)
        return cse_out

class LateContextualization(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layers = [5,7,9,11]
        self.graph_transformers = nn.ModuleList([GraphTransformerLayer() for _ in range(8)])
        self.cross_segment_encoders = nn.ModuleList([CrossSegmentEncoderLayer(id) for id in self.bert_layers])
    
    def forward(self, bbg, attention_mask, node_split, edge_split):
        bbg = self.graph_transformers[0](bbg)
        bbg = self.graph_transformers[1](bbg)
        bbg = self.graph_transformers[2](bbg)
        bbg = self.graph_transformers[3](bbg)
        bbg = self.graph_transformers[4](bbg)
        bbg, cse_out = self.cross_segment_encoders[0](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[5](bbg)
        bbg, cse_out = self.cross_segment_encoders[1](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[6](bbg)
        bbg, cse_out = self.cross_segment_encoders[2](bbg, attention_mask, node_split, edge_split)
        bbg = self.graph_transformers[7](bbg)
        bbg, cse_out = self.cross_segment_encoders[3](bbg, attention_mask, node_split, edge_split)
        return cse_out

class AH(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layers = [8,9,10,11]
        self.graph_transformers = nn.ModuleList([GraphTransformerLayer() for _ in range(8)])
        self.cross_segment_encoders = nn.ModuleList([CrossSegmentEncoderLayer(id) for id in self.bert_layers])
    
    def forward(self, bbg, attention_mask, node_split, edge_split):
        bbg = self.graph_transformers[0](bbg)
        bbg = self.graph_transformers[1](bbg)
        bbg = self.graph_transformers[2](bbg)
        bbg = self.graph_transformers[3](bbg)
        bbg = self.graph_transformers[4](bbg)
        bbg = self.graph_transformers[5](bbg)
        bbg = self.graph_transformers[6](bbg)
        bbg = self.graph_transformers[7](bbg)
        bbg, cse_out = self.cross_segment_encoders[0](bbg, attention_mask, node_split, edge_split)
        bbg, cse_out = self.cross_segment_encoders[1](bbg, attention_mask, node_split, edge_split)
        bbg, cse_out = self.cross_segment_encoders[2](bbg, attention_mask, node_split, edge_split)
        bbg, cse_out = self.cross_segment_encoders[3](bbg, attention_mask, node_split, edge_split)
        return cse_out