import os
os.environ["DGLBACKEND"] = "pytorch"

import re
import dgl
import json
import torch
import spacy
import amrlib
import penman
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
from transformers import AutoModel
from data.flota import FlotaTokenizer

MIN_LEN = 5
MAX_LEN = 100 # max length of a sentence
MAX_SENT = 100 # max number of sentences in a case
NUM_LABELS = 10 # from ecthr dataset
VOCAB_SIZE = 5000

color_ref = {"train":"red", 
             "validation":"blue", 
             "test":"yellow"}

nlp = spacy.load("en_core_web_lg")
stog = amrlib.load_stog_model(batch_size = 24)
ecthr_dataset = load_dataset("coastalcph/fairlex", "ecthr")

def get_sent_length(sent: str):
    return len(sent.split())

def preprocess_case(legal_case, return_length=False):
    """split sentences
       truncate both sentences and number of sentences"""
    sent_len = []
    sent_cnt = 0
    sentences = []
    for sent in nlp(legal_case).sents:
        if sent_cnt > MAX_SENT and not return_length:
            break # truncate the case
        sent = sent.text
        length = get_sent_length(sent)
        if length < MIN_LEN:
            continue 
        if length > MAX_LEN:
            # truncate the sentence
            sent = " ".join(sent.split()[:MAX_LEN])
        sent_len.append(length)
        sentences.append(sent)
        sent_cnt += 1

    if return_length:
        return sentences, sent_len
    else:
        return sentences

@torch.no_grad()
def count_case(dataset):
    """count number of sentences in cases and sentence length"""
    fig, axs = plt.subplots(2,1, figsize=(10, 6))
    for split in ["train", "validation", "test"]:
        split_dataset = dataset[split]
        sent_len = [] # record sentence Length
        sent_cnt = [] # record No. sentences in a case
        for legal_case in split_dataset["text"]:
            sentences, len_list = preprocess_case(legal_case, True)
            sent_len.extend(len_list)
            sent_cnt.append(len(sentences))

        sent_len_counter = sorted(Counter(sent_len).most_common())
        sent_cnt_counter = sorted(Counter(sent_cnt).most_common())
        # visualization
        axs[0].hist([t[0] for t in sent_len_counter],
                    weights = np.sqrt([t[1] for t in sent_len_counter]),
                    bins = 100, density = True, alpha = 0.5,
                    label = split, color = color_ref.get(split))
        axs[1].hist([t[0] for t in sent_cnt_counter],
                    weights = [t[1] for t in sent_cnt_counter],
                    bins = 100, density = True, alpha = 0.5,
                    label = split, color = color_ref.get(split))

    axs[0].set_xlim(xmin=0, xmax=700)
    axs[0].set_xlabel('Length', labelpad=-11, x=1.05)
    axs[0].set_ylabel('sqrt{frequency}')
    axs[0].legend(frameon=False)
    axs[1].set_xlim(xmin=0, xmax=700)
    axs[1].set_xlabel('Count', labelpad=-11, x=1.05)
    axs[1].set_ylabel('frequency')
    axs[1].legend(frameon=False)
    plt.show()
    fig.savefig("stat.svg", dpi=300, format="svg")
            
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
def sent2graph(dataset):
    vocab_table = torch.load("data/vocab_table.pt")
    with open("data/edge_table.json") as f:
        edge_table = json.load(f)

    for split in ["train","validation", "test"]:
        split_dataset = dataset[split]
        graph_x = [] # batched graphs of cases
        graph_y = label2onehot(split_dataset["labels"])
        for legal_case in split_dataset["text"]:
            sentences = preprocess_case(legal_case)
            amr_graphs = stog.parse_sents(sentences)
            sent_graph = [] # graphs for sentences in a case
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
                # save top node index
                top_index = node_ref.get(penman_graph.top)
                top = torch.zeros(dgl_graph.num_nodes())
                top[top_index] = 1
                # assign features to dgl graph
                dgl_graph.ndata["top_mask"] = top
                dgl_graph.ndata["node_ids"] = torch.LongTensor(node_ids)
                dgl_graph.edata["edge_ids"] = torch.LongTensor(edge_ids)
                # to undirected
                dgl_graph = dgl.add_reverse_edges(dgl_graph, copy_edata=True)
                sent_graph.append(dgl_graph)   

            graph_x.append(dgl.batch(sent_graph))
        # save to disk
        graph_dataset = {"x" : graph_x, "y" : graph_y}
        torch.save(graph_dataset, f"{split}_graph_dataset.pt")

def process_word(word):
    if "-" in word:
        word = re.sub(r"-\d+$", "", word) # remove digits
        word = " ".join(word.split("-"))

    return word
    
@torch.no_grad()   
def get_initial_embedding_weight():
    """get initial node embedding from bert using FLOTA"""
    bert = AutoModel.from_pretrained("bert-base-uncased", 
                                      output_hidden_states=True)
    tokenizer = FlotaTokenizer("bert-base-uncased", k = 3)
    vocab_table = torch.load("data/vocab_table.pt")

    initial_node_embedding = []
    initial_node_embedding.append(torch.randn(768)) # "other" -> 0
    for word in vocab_table.keys():
        word = process_word(word)
        input = tokenizer(word, return_tensors="pt")
        output = bert(**input)
        embedding_output = output[2][0]
        # average
        vocab_embedding = torch.mean(embedding_output[0][1:-1], dim=0)
        initial_node_embedding.append(vocab_embedding)
    
    initial_node_embedding = torch.stack(initial_node_embedding, dim=0)
    torch.save(initial_node_embedding, "initial_node_embedding.pt")
    torch.save(bert.embeddings.position_embeddings.weight, "initial_pos_embedding.pt")