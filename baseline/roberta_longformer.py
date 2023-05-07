import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (RobertaModel, RobertaTokenizer, LongformerForSequenceClassification,
                          TrainingArguments, Trainer, EvalPrediction)
from transformers.models.longformer.modeling_longformer import (LongformerSelfAttention, LongformerIntermediate, 
                                                                LongformerOutput, LongformerSelfOutput)
from datasets import load_dataset
import wandb

wandb.init(
    project = "roberta_longformer",
    config={
        "learning_rate": 1e-5,
        "architecture": "Longformer",
        "dataset": "ECtHR",
        "epochs": 20,
    }
)

lf = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096", 
    num_labels = 11, 
    problem_type="multi_label_classification")
roberta = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = load_dataset("coastalcph/fairlex", 'ecthr')

# apply weights from legal-bert to longformer
with torch.no_grad():

    max_pos = 4096
    config = lf.config

    # with legal-bert tokenizer, NOT longformer
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos

    '''
        Embeddings:
            word_embeddings:
            position_embeddings:
            token_type_embeddings: default
            LayerNorm:
    '''
    # change word_embeddings
    config.vocab_size = 30522 # vocab_size of legal-bert
    new_word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    new_word_embed.weight = nn.Parameter(roberta.embeddings.word_embeddings.weight)
    lf.longformer.embeddings.word_embeddings = new_word_embed
    
    # extend position embeddings
    max_pos += 2
    embed_size = roberta.embeddings.position_embeddings.weight.shape[1]
    new_pos_embed = torch.empty((max_pos, embed_size))
    # copy position embeddings over and over to initialize the new position embeddings
    ## Leave the positional embeddings at indices 0 and 1 unchanged
    new_pos_embed[0:2] = lf.longformer.embeddings.position_embeddings.weight[0:2]
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = 512
    while k < max_pos-1:
        new_pos_embed[k:(k + step)] = roberta.embeddings.position_embeddings.weight[2:]
        k += step

    lf.longformer.embeddings.position_embeddings.weight = nn.Parameter(new_pos_embed)

    # LayerNorm
    lf.longformer.embeddings.LayerNorm.weight = nn.Parameter(roberta.embeddings.LayerNorm.weight)
    lf.longformer.embeddings.LayerNorm.bias = nn.Parameter(roberta.embeddings.LayerNorm.bias)

    '''
        Each layer:
            attention: local/global attention, attention output
            intermediate: ...
            output ...
    '''
    for i, (lf_layer, bert_layer) in enumerate(zip(lf.longformer.encoder.layer, roberta.encoder.layer)):
        # attention
        longformer_self_attn = LongformerSelfAttention(config, layer_id = i)
        ## local attention
        longformer_self_attn.query = bert_layer.attention.self.query
        longformer_self_attn.key = bert_layer.attention.self.key
        longformer_self_attn.value = bert_layer.attention.self.value
        ## global attention
        longformer_self_attn.query_global = copy.deepcopy(bert_layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(bert_layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(bert_layer.attention.self.value)

        lf_layer.attention.self = longformer_self_attn

        ## attention output
        longformer_self_output = LongformerSelfOutput(config)
        longformer_self_output.dense = bert_layer.attention.output.dense
        longformer_self_output.LayerNorm = bert_layer.attention.output.LayerNorm

        lf_layer.attention.output = longformer_self_output

        # intermediate
        longformer_inter = LongformerIntermediate(config)
        longformer_inter.dense = bert_layer.intermediate.dense
        lf_layer.intermediate = longformer_inter

        # output
        longformer_output = LongformerOutput(config)
        longformer_output.dense = bert_layer.output.dense
        longformer_output.LayerNorm = bert_layer.output.LayerNorm
        lf_layer.output = longformer_output

# pre-processing dataset
def tokenize_function(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_matrix = np.zeros((len(text), 11))
    for idx, label_list in enumerate(examples['labels']):
        if len(label_list) == 0:
            labels_matrix[idx, 10] = 1.0
        else:
            for label in label_list:
                labels_matrix[idx, label] = 1.0
    encoding["vectorized_label"] = labels_matrix.tolist()
    return encoding

tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True,
                                remove_columns=dataset['train'].column_names)

tokenized_dataset.set_format('torch')
train_dataset = tokenized_dataset["train"].rename_column("vectorized_label", "labels")
val_dataset = tokenized_dataset["validation"].rename_column("vectorized_label", "labels")
test_dataset = tokenized_dataset["test"].rename_column("vectorized_label", "labels")

# training and evaluation
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


batch_size = 4
metric_name = "f1"
args = TrainingArguments(
    f"roberta-longformer",
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
    lf,
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