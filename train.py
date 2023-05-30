import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import argparse
import numpy as np
import wandb
import os
os.environ['DGLBACKEND'] = 'pytorch' 

from data.dataset import LegalDataset
from layer.model import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default=None, type=int,
                        help="Max number of training epochs to perform.")
    parser.add_argument("--layout", default=None, type=str, 
                        help="i, ec, lc, ah")
    parser.add_argument("--batch_size", default=None, type=int, 
                        help="batch size")
    parser.add_argument("--lr", default=1e-5, type=float, 
                        help="learning rate")
    return parser.parse_args()

def apply_threshold(logits, threshold=0.5):
    logits[logits  > threshold] = 1
    logits[logits <= threshold] = 0
    return logits

@torch.no_grad()
def evaluate(model, dataloader, criterion, mode, args):
    model.eval()
    all_logits, all_labels = [], []
    all_loss = []
    for graphs, labels in dataloader:
        logits = model(graphs).to("cpu")
        loss = criterion(logits, labels.float()).item()
        # record
        all_logits.append(logits)
        all_labels.append(labels)
        all_loss.append(loss)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_pred = apply_threshold(all_logits)
    all_labels = torch.cat(all_labels, dim=0)

    micro_f1 = f1_score(all_labels, all_pred, average="micro")
    accuracy = accuracy_score(all_labels, all_pred)
    roc_auc = roc_auc_score(all_labels, all_pred, average="micro")
    loss = np.mean(all_loss)

    if mode == "validation":
        return micro_f1, loss
    elif mode == "testing":
        # results on each class
        each_f1 = f1_score(all_labels, all_pred, average=None)
        np.save(f"data/{args.layout}-each-class-f1", each_f1)
        # gender fairness
        torch.save(all_pred, f"data/{args.layout}-prediction.pt")
        print(f"micro f1 score: {micro_f1}, loss: {loss}, roc_auc:{roc_auc}, accuracy:{accuracy}")
    else:
        raise ValueError("Invalid mode.")

def graph_collate_fn(batch):
    graphs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0)
    return graphs, labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()
    torch.manual_seed(0)
    wandb.init(
        project = "legal judgment prediction",
        config = {
            "architecture": args.layout,
            "learning_rate": args.lr,
            "dataset": "ECtHR",
            "epochs": args.num_epoch,
    })

    model = Net(args.layout).to(DEVICE)
    print(f"{args.layout} Params: {count_parameters(model)}")
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), args.lr)

    train_dataset = LegalDataset("train")
    val_dataset = LegalDataset("validation")
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  collate_fn=graph_collate_fn)
    best_micro_f1 = 0
    for _ in range(args.num_epoch):
        # training
        model.train()
        train_loss = []
        for graphs, labels in train_dataloader:
            optim.zero_grad()
            logits = model(graphs).to("cpu")
            loss = criterion(logits, labels.float())
            train_loss.append(loss.item())
            loss.backward()
            optim.step()
        
        train_loss = np.average(train_loss)
        # evaluation
        val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size, 
                            collate_fn=graph_collate_fn)
        micro_f1, val_loss = evaluate(model, val_dataloader, criterion, "validation", args)
        wandb.log({"train_loss": train_loss, "val_micro_f1": micro_f1, "val_loss": val_loss})
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            torch.save(model.state_dict(), f"{args.layout}-best-model-parameters.pt")
    # testing
    model.load_state_dict(torch.load(f"{args.layout}-best-model-parameters.pt"))
    test_dataset = LegalDataset("test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size, 
                                 collate_fn=graph_collate_fn)
    evaluate(model, test_dataloader, criterion, "testing", args) 
    

if __name__ == '__main__':
    main()