import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os

TRAIN_INPUT_CSV = "/root/VQPP/resources/GRAM/metrics/msrvtt_train.csv"
VAL_INPUT_CSV   = "/root/VQPP/resources/GRAM/metrics/msrvtt_val.csv"

MODEL_PATH = "/root/VQPP/baselines/finetune_bert/best_model.pth"
GT_RECALL_THRESHOLD = 0.1
BATCH_SIZE = 128

OUTPUT_TRAIN_CSV = "/root/VQPP/rephrasing/train_dpo.csv"
OUTPUT_VAL_CSV   = "/root/VQPP/rephrasing/validation_dpo.csv"

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

def score_queries(df, model, tokenizer, device):
    """Helper function to score a dataframe of queries."""
    print(f"Scoring {len(df)} queries...")
    all_scores = []
    queries = df['Query'].astype(str).tolist()

    model.eval()
    for i in tqdm(range(0, len(queries), BATCH_SIZE)):
        batch_texts = queries[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            scores = outputs.flatten().cpu().numpy()
            all_scores.extend(scores)
    
    return all_scores

def process_and_save():
    # 1. Setup Device & Model
    print("Initializing custom BertRegressor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertRegressor(n_outputs=1)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    if os.path.exists(TRAIN_INPUT_CSV):
        print(f"\nProcessing TRAINING data: {TRAIN_INPUT_CSV}")
        train_df = pd.read_csv(TRAIN_INPUT_CSV)
        
        train_df['predicted_score'] = score_queries(train_df, model, tokenizer, device)

        pred_threshold = np.percentile(train_df['predicted_score'], 50)
        print(f"Train Median Score Threshold: {pred_threshold:.4f}")

        filtered_train = train_df[
            (train_df['Recall@10'] < GT_RECALL_THRESHOLD) & 
            (train_df['predicted_score'] < pred_threshold)
        ].copy()

        filtered_train = filtered_train.sort_values(by="predicted_score", ascending=True)
        filtered_train = filtered_train.head(15000)

        filtered_train = filtered_train[['Query', 'Recall@10', 'predicted_score']].rename(columns={
            'Query': 'query', 'Recall@10': 'ground_truth_recall', 'predicted_score': 'initial_score'
        })

        filtered_train.to_csv(OUTPUT_TRAIN_CSV, index=False)
        print(f"Saved TRAIN set ({len(filtered_train)} samples) to: {OUTPUT_TRAIN_CSV}")
    else:
        print(f"Warning: Training file not found at {TRAIN_INPUT_CSV}")

    if os.path.exists(VAL_INPUT_CSV):
        print(f"\nProcessing VALIDATION data: {VAL_INPUT_CSV}")
        val_df = pd.read_csv(VAL_INPUT_CSV)
        
        val_df['predicted_score'] = score_queries(val_df, model, tokenizer, device)

        filtered_val = val_df[
            (val_df['Recall@10'] < GT_RECALL_THRESHOLD)
        ].copy()

        filtered_val = filtered_val[['Query', 'Recall@10', 'predicted_score']].rename(columns={
            'Query': 'query', 'Recall@10': 'ground_truth_recall', 'predicted_score': 'initial_score'
        })

        filtered_val.to_csv(OUTPUT_VAL_CSV, index=False)
        print(f"Saved VALIDATION set ({len(filtered_val)} samples) to: {OUTPUT_VAL_CSV}")
    else:
        print(f"Warning: Validation file not found at {VAL_INPUT_CSV}")

if __name__ == "__main__":
    process_and_save()