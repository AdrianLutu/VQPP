import os
import time
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import scipy.stats
import json
from groq import Groq
from api_key import *

def calculate_correlations(list1, list2):
    if len(list1) != len(list2):
        return "The lists are not of the same length"

    pearson_corr, pvaluep = scipy.stats.pearsonr(list1, list2)

    kendall_corr, pvalue = scipy.stats.kendalltau(list1, list2)

    return pearson_corr, pvaluep, kendall_corr, pvalue

RETRIEVAL_MODEL = "VAST"  # Options: "GRAM", "VAST"
DATASET = "vatex"  # Options: "msrvtt", "vatex"
METRIC = "Recall@10"  # Options: "Recall@10", "Reciprocal_Rank"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(REPO_ROOT, "resources", RETRIEVAL_MODEL, "metrics")

# The in-context examples are the nearest training queries in the space of pre-trained
# BERT embeddings, as described in the paper.
EMBEDDING_MODEL = "bert-base-cased"
BATCH_SIZE = 128
TOPK= 16

# How each metric is described to the LLM.
METRIC_DESCRIPTIONS = {
    "Recall@10": "the 'Recall@10' (1.0 if the correct video is ranked in the top 10, 0.0 otherwise)",
    "Reciprocal_Rank": "the 'Reciprocal Rank' (1 / rank of the correct video, a float between 0.0 and 1.0)",
}

groq_client = Groq(api_key= API_KEY_GROQ)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODEL)
bert_model = BertModel.from_pretrained(EMBEDDING_MODEL).to(device)
bert_model.eval()

def get_embeddings(text_list):
    """Mean-pooled last hidden states of pre-trained BERT, one row per text.

    The padding tokens are left out of the mean through the attention mask. Mean pooling
    is used rather than the [CLS] state because the [CLS] state of a BERT that was not
    fine-tuned is a poor sentence representation for nearest-neighbour search.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="Embedding Batches"):
        batch = text_list[i : i + BATCH_SIZE]
        inputs = bert_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            hidden_states = bert_model(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def predict_score_with_groq(val_query, similar_examples):
    """
    Predicts METRIC using Groq (Llama 3.1 8B).
    """
    prompt = "You are a ranking prediction system.\n"
    prompt += f"Predict {METRIC_DESCRIPTIONS[METRIC]} for the Target Query based on the provided Reference Examples.\n\n"

    prompt += "Reference Examples:\n"
    for ex in similar_examples:
        prompt += f"- Query: \"{ex['query']}\" | {METRIC}: {ex['score']}\n"

    prompt += f"\nTarget Query: \"{val_query}\"\n"
    prompt += "Output strictly in JSON format: {\"predicted_score\": <number>}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0,
        )

        response_content = chat_completion.choices[0].message.content
        print("Got Response")
        return response_content

    except Exception as e:
        print(f"Groq Error: {e}")
        return None

def main():
    df_train = pd.read_csv(os.path.join(METRICS_DIR, f"{DATASET}_train.csv"))
    df_val = pd.read_csv(os.path.join(METRICS_DIR, f"{DATASET}_test.csv"))

    # Ground truth is collected next to the predictions, inside the loop. Taking the
    # whole column up front would leave the two lists shifted as soon as a query is
    # skipped, and calculate_correlations would then return a string instead of the
    # four values the caller unpacks.
    scores = []

    train_queries = df_train['Query'].astype(str).tolist()

    train_embeddings = get_embeddings(train_queries)

    nn_model = NearestNeighbors(n_neighbors=TOPK, metric='cosine')
    nn_model.fit(train_embeddings)

    val_queries = df_val['Query'].astype(str).tolist()
    val_embeddings = get_embeddings(val_queries)

    results = []
    for index, row in df_val.iterrows():
        val_query = str(row['Query'])
        val_emb = val_embeddings[index].reshape(1, -1)

        distances, indices = nn_model.kneighbors(val_emb)

        similar_examples = []
        for idx in indices[0]:
            similar_examples.append({
                'query': df_train.iloc[idx]['Query'],
                'score': df_train.iloc[idx][METRIC]
            })

        json_response = predict_score_with_groq(val_query, similar_examples)

        final_score = 0.0
        if json_response:
            try:
                data = json.loads(json_response)
                final_score = float(data.get("predicted_score", 0.0))
                print(f"Query: {val_query} | Predicted: {final_score}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Query: {val_query} | Parse Error: {e}")
                import re
                match = re.search(r"0\.\d+", json_response)
                if match:
                    final_score = float(match.group())
                    print(f"  -> Recovered via Regex: {final_score}")
        else:
            print(f"Query: {val_query} | Failed to get response")

        results.append(final_score)
        scores.append(row[METRIC])

        time.sleep(4)

    pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(
    scores, results
    )
    print(pearson_corr, pvaluep, kendall_corr, pvalue)

if __name__ == "__main__":
    main()
