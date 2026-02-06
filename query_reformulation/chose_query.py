import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import re
import csv

REWARD_MODEL_PATH = "/root/VQPP/baselines/finetune_bert/best_model.pth"
CSV_INPUT_PATH = '/root/VQPP/rephrasing/phi-4-reformulator-final/test_rephrased_step_400.csv'
CSV_OUTPUT_PATH = '/root/VQPP/rephrasing/benchmarked_queries_step400.csv'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.linear2 = nn.Linear(512, n_outputs)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

def load_reward_model(path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertRegressor()
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except:
        model = torch.load(path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def get_query_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines[1:]:
        match = re.search(r',(\d+\.\d+|\d+),(\d+),(\d+),(\d+),(\d+),', line)
        if match:
            start_index = match.start()
            end_index = match.end()
            orig = line[:start_index].strip().strip('"')
            rephrased = line[end_index:].strip().strip('"').replace('""', '"')
            pairs.append({'original': orig, 'rephrased': rephrased})
    return pairs

def get_score(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    return outputs.squeeze().cpu().tolist()

def main():
    print(f"Loading model from {REWARD_MODEL_PATH}...")
    tokenizer, model = load_reward_model(REWARD_MODEL_PATH)
    
    print("Parsing CSV...")
    data = get_query_pairs(CSV_INPUT_PATH)
    
    results = []
    print(f"Benchmarking {len(data)} pairs...")

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        orig_texts = [b['original'] for b in batch]
        reph_texts = [b['rephrased'] for b in batch]
        
        orig_scores = get_score(orig_texts, tokenizer, model)
        reph_scores = get_score(reph_texts, tokenizer, model)
        
        if isinstance(orig_scores, float):
            orig_scores = [orig_scores]
            reph_scores = [reph_scores]

        for j in range(len(batch)):
            orig_s = orig_scores[j]
            reph_s = reph_scores[j]
            
            is_rephrased_better = reph_s > orig_s
            best_query = reph_texts[j] if is_rephrased_better else orig_texts[j]
            
            results.append({
                'original_query': orig_texts[j],
                'rephrased_query': reph_texts[j],
                'original_score': orig_s,
                'rephrased_score': reph_s,
                'selected_query': best_query,
                'was_rephrased': is_rephrased_better
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(CSV_OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"Done! Results saved to {CSV_OUTPUT_PATH}")
    print(f"Rephrased version was better in {df_results['was_rephrased'].sum()} out of {len(df_results)} cases.")

if __name__ == "__main__":
    main()