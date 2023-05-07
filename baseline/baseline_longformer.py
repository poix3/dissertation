import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from datasets import load_dataset
import wandb

wandb.init(
    project = "baseline-longformer",
    config={
        "learning_rate": 1e-5,
        "architecture": "Longformer",
        "dataset": "ECtHR",
        "epochs": 20,
    }
)

model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    num_labels=11,
    problem_type="multi_label_classification",
)
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# pre-processing dataset
def tokenize_function(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_matrix = np.zeros((len(text), 11))
    for idx, label_list in enumerate(examples["labels"]):
        if len(label_list) == 0:
            labels_matrix[idx, 10] = 1.0
        else:
            for label in label_list:
                labels_matrix[idx, label] = 1.0
    encoding["vectorized_label"] = labels_matrix.tolist()
    return encoding


dataset = load_dataset("coastalcph/fairlex", "ecthr")
tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)
tokenized_dataset.set_format("torch")
train_dataset = tokenized_dataset["train"].rename_column("vectorized_label", "labels")
val_dataset = tokenized_dataset["validation"].rename_column("vectorized_label", "labels")
test_dataset = tokenized_dataset["test"].rename_column("vectorized_label", "labels")

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


# training and evaluation
batch_size = 4
metric_name = "f1"
args = TrainingArguments(
    f"longformer-baseline",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to="wandb",
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
print("-----training finished and best model loaded-----")
metrics = trainer.predict(test_dataset)
print(metrics)