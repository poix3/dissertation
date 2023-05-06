import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import spacy
import amrlib
import penman
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
from transformers import AutoModel
from data.flota import FlotaTokenizer

MIN_LEN = 5
MAX_LEN = 500
NUM_LABELS = 10 # from ecthr dataset
VOCAB_SIZE = 5000

nlp = spacy.load("en_core_web_lg")
stog = amrlib.load_stog_model(batch_size = 32)
ecthr_dataset = load_dataset("coastalcph/fairlex", "ecthr")

def get_sent_length(sent: str):
    return len(sent.split())

def preprocess_case(legal_case, return_length=False):
    """split sentences and truncate"""
    sent_len = []
    sentences = []
    for sent in nlp(legal_case).sents:
        sent = sent.text
        length = get_sent_length(sent)
        if length < MIN_LEN:
            continue 
        if length > MAX_LEN:
            sent = " ".join(sent.split()[:MAX_LEN])
        sent_len.append(length)
        sentences.append(sent)

    if return_length:
        return sentences, sent_len
    else:
        return sentences

def count_case(dataset):
    """count number of sentences in cases and sentence length"""
    for split in ["train", "validation", "test"]:
        split_dataset = dataset[split]
        sent_len = [] # record sentence Length
        sent_cnt = [] # record No. sentences in a case
        for legal_case in split_dataset["text"]:
            sentences, len_list = preprocess_case(legal_case, True)
            sent_len.extend(len_list)
            sent_cnt.append(len(sentences))
        # visualization TODO
        sent_len_counter = sorted(Counter(sent_len).most_common())
        sent_cnt_counter = sorted(Counter(sent_cnt).most_common())

            

def label2onehot(labels):
    label_tensor = torch.zeros(len(labels), 
                               NUM_LABELS, 
                               dtype=torch.long)
    for i, label in enumerate(labels):
        if label:
            label_tensor[i, torch.tensor(label)] = 1
    return label_tensor

@torch.no_grad()
def record_and_save(dataset, vocab_size = VOCAB_SIZE):

    node_label = [] # e.g. "want-01"
    edge_label = [] # e.g. ":ARG0"
    cons_label = [] # e.g. ":quant", ":polarity"

    for split in ["train","validation", "test"]:
        split_dataset = dataset[split]
        for legal_case in split_dataset["text"]:
            sentences = preprocess_case(legal_case)            
            amr_graphs = stog.parse_sents(sentences)
            for amr_graph in amr_graphs:
                if not isinstance(amr_graph, str):
                    continue
                penman_graph = penman.decode(amr_graph)
                node_label.extend([instance.target 
                                   for instance in penman_graph.instances()])
                edge_label.extend([edge.role 
                                   for edge in penman_graph.edges()]) 
                cons_label.extend([attribute.role 
                                   for attribute in penman_graph.attributes()])
                
    # save to disk
    vocab = [item[0] for item in Counter(node_label).most_common(vocab_size)]
    vocab_table = {key: i for i, key in enumerate(vocab, 1)}
    torch.save(vocab_table, "vocab_table.pt")
    edge_label = Counter(edge_label).most_common()
    torch.save(edge_label, "edge_label.pt")
    cons_label = Counter(cons_label).most_common()
    torch.save(cons_label, "cons_label.pt")

@torch.no_grad()
def sent2graph(dataset, split):
    vocab_table = torch.load("vocab_table.pt")
    # TODO add edge and constant label
    edge_table = {
        ":ARG0" : 1,
        ":ARG1" : 2,
        ":ARG2" : 3,
        ":ARG3" : 4,
        "ARG1-of" : 5, 
    } 

    for split in ["train","validation", "test"]:
        split_dataset = dataset[split]
        graph_x = []
        graph_y = label2onehot(split_dataset["labels"])
        for legal_case in split_dataset["text"]:
            sentences = preprocess_case(legal_case)
            amr_graphs = stog.parse_sents(sentences)
            for amr_graph in amr_graphs:
                if not isinstance(amr_graph, str):
                    continue
                penman_graph = penman.decode(amr_graph)
                # penman graph to DGL graph
                node_ref = {} # "other" -> 0
                node_ids, edge_ids = [], []
                src, dst = [], []

                for i, instance in enumerate(penman_graph.instances()):
                    node_ref[instance.source] = i
                    node_ids.append(vocab_table.get(instance.target, 0))
                    
                for edge in penman_graph.edges():
                    src.append(node_ref.get(edge.source))
                    dst.append(node_ref.get(edge.target)) 
                    edge_ids.append(edge_table.get(edge.role, 0))

                dgl_graph = dgl.graph((src, dst), num_nodes=len(node_ref))
                # to undirected
                dgl_graph = dgl.to_bidirected(dgl_graph)
                edge_ids = edge_ids * 2
                # save top node index
                top_index = node_ref.get(penman_graph.top)
                top = torch.zeros(dgl_graph.num_nodes())
                top[top_index] = 1
                # assign features to dgl graph
                dgl_graph.ndata["top"] = top
                dgl_graph.ndata["node_ids"] = torch.LongTensor(node_ids)
                dgl_graph.edata["edge_ids"] = torch.LongTensor(edge_ids)
                graph_x.append(dgl_graph)   
        # save to disk
        graph_dataset = {"x" : graph_x, "y" : graph_y}
        torch.save(graph_dataset, f"{split}_graph_dataset.pt")
            
def get_initial_embedding_weight():
    """get initial node embedding from bert using FLOTA"""
    model = AutoModel.from_pretrained("bert-base-uncased", 
                                      output_hidden_states=True)
    tokenizer = FlotaTokenizer("bert-base-uncased", k = 3)
    vocab_table = torch.load("vocab_table.pt")

    initial_embedding = []
    initial_embedding.append(torch.randn(768)) # "other" -> 0
    for vocab in vocab_table.keys():
        words = " ".join(vocab.split("-")[:-1])
        input = tokenizer(words, return_tensors="pt")
        output = model(**input)
        embedding_output = output[2][0]
        # average
        vocab_embedding = torch.mean(embedding_output[0][1:-1], dim=0)
        initial_embedding.append(vocab_embedding)
    
    initial_embedding = torch.stack(initial_embedding, dim=0)
    torch.save(initial_embedding, "initial_embedding.pt")