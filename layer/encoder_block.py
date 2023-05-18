import dgl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from layer.graphTransformer_layer import GraphTransformerLayer

class Encoder(nn.Module):
    def __init__(self, id, in_dim=768, out_dim=768, num_heads=4):
        super().__init__()
        self.graph_encoder = GraphTransformerLayer(in_dim, out_dim, num_heads)
        self.cross_segment_encoder = self.apply_bert_weight(id)

    @torch.no_grad()
    def apply_bert_weight(self, id):
        bert = BertModel.from_pretrained("bert-base-uncased")
        return bert.encoder.layer[id]

    def get_extended_attention_mask(self, attention_mask, dtype=torch.float):
        """https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/modeling_utils.py#L873-L895"""
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, bbg, attention_mask, node_split, edge_split):
        """ bbg: b  - num traning sample / batch size
                 bg - batched graph from a case """
        bbg = self.graph_encoder(bbg)
        bgs = dgl.unbatch(bbg, node_split, edge_split) # list
        # prepare data for CSE
        batched_top_feat = []
        for bg in bgs:
            top_mask = bg.ndata["top_mask"]
            top_feat = bg.ndata["node_feat"][top_mask == 1] # dim: [num_top_nodes, feat_dim]
            batched_top_feat.append(top_feat)

        batched_top_feat = pad_sequence(batched_top_feat, batch_first=True) # dim: [batch_size, max_num_top_nodes, feat_dim]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        batched_top_feat = self.cross_segment_encoder(batched_top_feat, extended_attention_mask)[0]
        # update top nodes
        for top_feat, att_mask, bg in zip(batched_top_feat, attention_mask, bgs):
            num_top = sum(att_mask == 1)
            top_feat = top_feat[:num_top] # unpad
            # update tops
            top_mask = bg.ndata["top_mask"]
            bg.ndata["node_feat"][top_mask == 1] = top_feat

        bbg = dgl.batch(bgs)
        return bbg, batched_top_feat # return batched_top_feat for head classification