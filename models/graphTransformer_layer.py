import torch
import torch.nn as nn
import numpy as np
import dgl.function as fn

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale_constant = np.sqrt(self.out_dim)
        self.W_Q = nn.Linear(in_dim, out_dim * num_heads)
        self.W_K = nn.Linear(in_dim, out_dim * num_heads)
        self.W_V = nn.Linear(in_dim, out_dim * num_heads)
        self.W_E = nn.Linear(in_dim, out_dim * num_heads)
        
    def propagate_attention(self, g):
        g.apply_edges(lambda edges: 
                      {"score": edges.src["K"] * edges.dst["Q"]})
        g.apply_edges(lambda edges: 
                      {"score": edges.data["score"] / self.scale_constant})
        g.apply_edges(lambda edges:
                      {"score": edges.data["score"] * edges.data["E"]})
        g.apply_edges(lambda edges:
                      {"e_out": edges.data["score"]})
        g.apply_edges(lambda edges:
                      {"score": torch.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5, 5))})
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V', 'score', 'V'), fn.sum('V', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
        
    def forward(self, g, h, e):
        # linear transformation
        Q = self.W_Q(h)
        K = self.W_K(h)
        V = self.W_V(h)
        E = self.W_E(e)
        # multi-head
        g.ndata["Q"] = Q.view(-1, self.num_heads, self.out_dim)
        g.ndata["K"] = K.view(-1, self.num_heads, self.out_dim)
        g.ndata["V"] = V.view(-1, self.num_heads, self.out_dim)
        g.edata["E"] = E.view(-1, self.num_heads, self.out_dim)
        # attention
        self.propagate_attention(g)
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out
    
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.O_h = nn.Sequential(nn.Dropout(self.dropout),
                                 nn.Linear(out_dim, out_dim))
        self.O_e = nn.Sequential(nn.Dropout(self.dropout),
                                 nn.Linear(out_dim, out_dim))
        self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        self.batch_norm2_h = nn.BatchNorm1d(out_dim)
        self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        self.attention = GraphAttentionLayer(in_dim, 
                                            out_dim//num_heads, 
                                            num_heads)             
        self.FFN_h = nn.Sequential(nn.Linear(out_dim, out_dim*2),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(out_dim*2, out_dim))
        self.FFN_e = nn.Sequential(nn.Linear(out_dim, out_dim*2),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(out_dim*2, out_dim))
        
    def forward(self, g):
        # node & edge features
        h = g.ndata["node_feat"]
        e = g.edata["edge_feat"]
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        # projection + residual connection 
        h = h + self.O_h(h_attn_out.view(-1, self.out_channels))
        e = e + self.O_e(e_attn_out.view(-1, self.out_channels))
        # batch norm
        h = self.batch_norm1_h(h)
        e = self.batch_norm1_e(e)
        # FFN + residual connection
        h = h + self.FFN_h(h) 
        e = e + self.FFN_e(e)
        # batch norm
        h = self.batch_norm2_h(h)
        e = self.batch_norm2_e(e)
        # update node & edge features
        g.ndata["node_feat"] = h
        g.edata["edge_feat"] = e

        return g           