import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import json
import spacy
import amrlib
import penman
from datasets import load_dataset
from collections import Counter

MIN_LEN = 5
MAX_LEN = 500

nlp = spacy.load("en_core_web_lg")
stog = amrlib.load_stog_model()
ecthr_dataset = load_dataset("coastalcph/fairlex", "ecthr")

def truncate(sent):
    words = sent.split()
    if len(words) > MAX_LEN:
        sent = ' '.join(words[:MAX_LEN])
    return sent


def count_length(dataset):
    for split in ["train", "validation", "test"]:
        sent_len = []
        split_dataset = dataset[split]
        for legal_case in split_dataset["text"]:
            for paragraph in legal_case.split("</s> "):
                sent_len.extend([
                    len(sent.text.split())
                    for sent in nlp(paragraph).sents
                    if len(sent.text) != 0
                ])
        # TODO Visualization


def create_ids(dataset):
    vocab = set()
    edge_label = []
    with torch.no_grad():
        for split in ["train","validation", "test"]:
            split_dataset = dataset[split]
            for legal_case in split_dataset["text"]:
                for paragraph in legal_case.split("</s> "):
                    sents = [truncate(sent.text)
                            for sent in nlp(paragraph).sents
                            if MIN_LEN < len(sent.text.split())]
                    graphs = stog.parse_sents(sents)
                    graphs = penman.iterdecode(" ".join(graphs))
                    for graph in graphs:
                        vocab.update([instance.target for instance in graph.instances()])
                        edge_label.extend([edge.role for edge in graph.edges()])

    vocab_table = {key: i for i, key in enumerate(vocab)}
    with open("vocab_table.json", "w") as f:
        json.dump(vocab_table, f)

    edge_label = Counter(edge_label).most_common()
    with open("edge_label.json", "w") as f:
        json.dump(edge_label, f)


def sent2graph(dataset, split):
    # load vocabulary table and choose edge labels
    with open('vocab_table.json', 'r') as f:
        vocab_table = json.load(f)
    edge_label = [] #TODO

    with torch.no_grad():
        for split in ["train","validation", "test"]:
            case_graphs = []
            dataset = dataset[split]
            for legal_case in dataset["text"]:
                para_graphs = []
                for paragraph in legal_case.split("</s> "):
                    sents = [truncate(sent.text)
                            for sent in nlp(paragraph).sents
                            if MIN_LEN < len(sent.text.split())]
                    serialized_graphs = stog.parse_sents(sents)
                    penman_graphs = penman.iterdecode(" ".join(serialized_graphs))
                    for g in penman_graphs:
                        
                        node_ref = {k: i for i, k in enumerate(g.variables())}
                        for e in g.edges():


                        dgl_graph = dgl.graph()