import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import LegalDataset
from layer.model import Net


if __name__ == "__main__":
    net = Net(num_layer=6, 
              num_e_labels=17)
    
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(net.parameters())
    
    train_dataset = LegalDataset("train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1, #TODO
                                  )
    # TODO 
