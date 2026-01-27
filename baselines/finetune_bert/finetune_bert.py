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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/metrics/msrvtt_train.csv")
df_test = pd.read_csv("/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/metrics/msrvtt_val.csv")

l = len(df)

df_train = df[["Query", "Recall@10"]].iloc[:int(0.2 * l)]
print(len(df_train))
df_eval = df.iloc[int(0.9 * l):]
df_test = df_test[["Query", "Recall@10"]]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_prepare(dataframe):
    inputs = tokenizer(
        list(dataframe["Query"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor(dataframe["Recall@10"].values).unsqueeze(-1).float()
    return inputs["input_ids"], inputs["attention_mask"], labels


train_inputs, train_masks, train_labels = tokenize_and_prepare(df_train)
eval_inputs, eval_masks, eval_labels = tokenize_and_prepare(df_eval)
test_inputs, test_masks, test_labels = tokenize_and_prepare(df_test)

test_scores_list = df_test["Recall@10"].tolist()

# +
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
eval_dataset = TensorDataset(eval_inputs, eval_masks, eval_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=256, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True)


# -


class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
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
            # Initialize model
            model = BertRegressor()
            model.to(device)

            # Initialize optimizer with current hyperparameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

            # Training loop
            for epoch in range(epochs):
                model.train()
                for i, batch in tqdm(enumerate(train_loader), total=len(train_loader),desc=f"Epoch {epoch}"):
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            mse_eval, r_squared_eval = evaluate_model(model, eval_loader)
            print(
                f"LR: {lr}, Epochs: {epochs}, Weight Decay: {decay}, Eval MSE: {mse_eval}, Eval R-squared: {r_squared_eval}"
            )

            # Update best model if current model is better
            if mse_eval < best_mse:
                best_mse = mse_eval
                best_params = {
                    "learning_rate": lr,
                    "num_epochs": epochs,
                    "weight_decay": decay,
                }
                # Save the best model
                torch.save(model.state_dict(), "best_model.pth")

# Print best parameters
print(f"Best Parameters: {best_params}")

# -

model = BertRegressor()
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
print("loaded")

mse_test, r_squared_test = evaluate_model(model, test_loader)
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r_squared_test}")


# +
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        all_predictions.extend(outputs.cpu().numpy())
# -

predictions_scores = [float(pred) for pred in all_predictions]


predictions_df = pd.DataFrame(predictions_scores, columns=["predicted_score"])
predictions_csv_path = "predicted_score_avg_scores_rr.csv"
predictions_df.to_csv(predictions_csv_path, index=False)


def calculate_correlations(list1, list2):
    # Check if the lists are of the same length
    if len(list1) != len(list2):
        return "The lists are not of the same length"

    # Calculate Pearson correlation
    pearson_corr, pvaluep = scipy.stats.pearsonr(list1, list2)

    # Calculate Kendall Tau correlation
    kendall_corr, pvalue = scipy.stats.kendalltau(list1, list2)

    return pearson_corr, pvaluep, kendall_corr, pvalue


pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(
    test_scores_list, predictions_scores
)
print(pearson_corr, pvaluep, kendall_corr, pvalue)