import dgl
import torch
from torch.utils.data import Dataset

class LegalDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.raw_dataset = torch.load(f"{split}_graph_dataset.pt")
        self.graphs = self.raw_dataset["x"]
        self.labels = self.raw_dataset["y"]

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.graphs[index], self.labels[index]