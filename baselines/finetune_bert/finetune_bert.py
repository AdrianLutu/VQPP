# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: PyTorch 1.13 (Local)
#     language: python
#     name: pytorch-1-13
# ---

# +
# pip install transformers

# +
# pip install -U kaleido
# -

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle
from transformers import BertModel, BertTokenizer
from torch.nn import Module, Linear, Dropout
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import scipy.stats
import pandas as pd
import pickle
from tqdm import tqdm


import numpy as np

import os

RETRIEVAL_MODEL = "GRAM"  # Options: "GRAM", "VAST"
DATASET = "vatex"  # Options: "msrvtt", "vatex"
METRIC = "Reciprocal_Rank"  # Options: "Reciprocal_Rank", "Recall@10"

# Number of epochs without an improvement of the validation MSE before a run stops.
PATIENCE = 3

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(REPO_ROOT, "resources", RETRIEVAL_MODEL, "metrics")

# Every configuration gets its own checkpoint, so the query reformulation scripts can load
# the predictor they need (GRAM, MSR-VTT, Recall@10) instead of whichever run came last.
RUN_NAME = f"bert_{RETRIEVAL_MODEL}_{DATASET}_{'rr' if METRIC == 'Reciprocal_Rank' else 'r10'}"
OUTPUT_DIR = os.path.join(REPO_ROOT, "baselines", "finetune_bert", "checkpoints")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.pth")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_test_predictions.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_train = pd.read_csv(os.path.join(METRICS_DIR, f"{DATASET}_train.csv"))
df_eval = pd.read_csv(os.path.join(METRICS_DIR, f"{DATASET}_val.csv"))
df_test = pd.read_csv(os.path.join(METRICS_DIR, f"{DATASET}_test.csv"))


df_train = df_train[["Query", METRIC]]
df_eval = df_eval[["Query", METRIC]]
df_test = df_test[["Query", METRIC]]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_prepare(dataframe):
    inputs = tokenizer(
        list(dataframe["Query"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor(dataframe[METRIC].values).unsqueeze(-1).float()
    return inputs["input_ids"], inputs["attention_mask"], labels


train_inputs, train_masks, train_labels = tokenize_and_prepare(df_train)
eval_inputs, eval_masks, eval_labels = tokenize_and_prepare(df_eval)
test_inputs, test_masks, test_labels = tokenize_and_prepare(df_test)

test_scores_list = df_test[METRIC].tolist()

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
eval_dataset = TensorDataset(eval_inputs, eval_masks, eval_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)


class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")

        # Only part of BERT is fine-tuned: the last encoder layer, the pooler and the
        # regression head. The other 11 encoder layers and the embeddings stay frozen.
        # The head reads the pooler output (the [CLS] state passed through BERT's
        # dense + tanh layer), not the raw [CLS] state. See the notes in the README.
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
            
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        self.drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.linear2 = nn.Linear(512, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

    def get_embeddings(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        x = self.linear1(pooled_output)
        return x


def evaluate_model(model, data_loader):
    model.eval()
    predictions, labels_list = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    predictions = np.array(predictions).flatten()
    labels_array = np.array(labels_list).flatten()

    mse = mean_squared_error(labels_array, predictions)
    r_squared = r2_score(labels_array, predictions)

    return mse, r_squared


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

param_grid = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "num_epochs": [12],
    "weight_decay": [0, 0.1, 0.01],
}

# +
best_mse = float("inf")
best_params = {}

for lr in param_grid["learning_rate"]:
    for epochs in param_grid["num_epochs"]:
        for decay in param_grid["weight_decay"]:
            model = BertRegressor()
            model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

            # Early stopping: the model is evaluated after every epoch, the best epoch is
            # the one kept, and the run stops once the validation MSE has not improved
            # for PATIENCE epochs.
            best_run_mse = float("inf")
            epochs_without_improvement = 0

            for epoch in range(epochs):
                model.train()
                for i, batch in tqdm(enumerate(train_loader), total=len(train_loader),desc=f"Epoch {epoch}"):
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()

                mse_eval, r_squared_eval = evaluate_model(model, eval_loader)
                print(
                    f"LR: {lr}, Epoch: {epoch + 1}/{epochs}, Weight Decay: {decay}, Eval MSE: {mse_eval}, Eval R-squared: {r_squared_eval}"
                )

                if mse_eval < best_run_mse:
                    best_run_mse = mse_eval
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if mse_eval < best_mse:
                    best_mse = mse_eval
                    best_params = {
                        "learning_rate": lr,
                        "num_epochs": epoch + 1,
                        "weight_decay": decay,
                    }

                    torch.save(model.state_dict(), CHECKPOINT_PATH)

                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping after epoch {epoch + 1}: no improvement for {PATIENCE} epochs.")
                    break

print(f"Best Parameters: {best_params}")

model = BertRegressor()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.to(device)
print("loaded")

mse_test, r_squared_test = evaluate_model(model, test_loader)
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r_squared_test}")

model.eval()
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        all_predictions.extend(outputs.cpu().numpy())

predictions_scores = [float(pred) for pred in all_predictions]


predictions_df = pd.DataFrame(predictions_scores, columns=["predicted_score"])
predictions_df.to_csv(PREDICTIONS_PATH, index=False)


def calculate_correlations(list1, list2):
    if len(list1) != len(list2):
        return "The lists are not of the same length"

    pearson_corr, pvaluep = scipy.stats.pearsonr(list1, list2)

    kendall_corr, pvalue = scipy.stats.kendalltau(list1, list2)

    return pearson_corr, pvaluep, kendall_corr, pvalue


pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(
    test_scores_list, predictions_scores
)
print(pearson_corr, pvaluep, kendall_corr, pvalue)