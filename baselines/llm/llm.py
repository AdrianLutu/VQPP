import os
import time
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import scipy
import json
from groq import Groq
from api_key import *

def calculate_correlations(list1, list2):
    if len(list1) != len(list2):
        return "The lists are not of the same length"

    pearson_corr, pvaluep = scipy.stats.pearsonr(list1, list2)

    kendall_corr, pvalue = scipy.stats.kendalltau(list1, list2)

    return pearson_corr, pvaluep, kendall_corr, pvalue

EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"
BATCH_SIZE = 100
RPM_LIMIT = 20
TOPK= 16

# Initialize Client
groq_client = Groq(api_key= API_KEY_GROQ)
client = genai.Client(api_key=API_KEY)

def get_embeddings_safe(text_list):
    all_embeddings = []
    total_items = len(text_list)
    
    sleep_interval = 60 / RPM_LIMIT + 1 

    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Embedding Batches"):
        batch = text_list[i : i + BATCH_SIZE]
        
        success = False
        retry_count = 0
        while not success and retry_count < 5:
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                
                batch_embs = [e.values for e in response.embeddings]
                all_embeddings.extend(batch_embs)
                success = True

            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (2 ** retry_count) * 10
                    print(f"\nRate limit hit. Cooling down for {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    print(f"\nCritical Error in batch starting at index {i}: {e}")
                    all_embeddings.extend([[0.0] * 768] * len(batch))
                    success = True

    return np.array(all_embeddings)

def predict_rank_with_groq(val_query, similar_examples):
    """
    Predicts rank using Groq (Llama 3.3) for higher daily limits.
    """
    prompt = "You are a ranking prediction system.\n"
    prompt += "Predict the 'Recall@10' (float 0.0 - 1.0) for the Target Query based on the provided Reference Examples.\n\n"
    
    prompt += "Reference Examples:\n"
    for ex in similar_examples:
        prompt += f"- Query: \"{ex['query']}\" | recall: {ex['recall']}\n"
    
    prompt += f"\nTarget Query: \"{val_query}\"\n"
    prompt += "Output strictly in JSON format: {\"predicted_recall\": <number>}"

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
    df_train = pd.read_csv('VAST\\vatex_train.csv')
    df_val = pd.read_csv('VAST\\vatex_val.csv')

    scores = df_val["Recall@10"].tolist()

    train_queries = df_train['Query'].astype(str).tolist()
    
    train_embeddings = get_embeddings_safe(train_queries)
    
    nn_model = NearestNeighbors(n_neighbors=TOPK, metric='cosine')
    nn_model.fit(train_embeddings)

    
    results = []
    for index, row in df_val.iterrows():
        val_query = str(row['Query'])
        
        try:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=val_query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            val_emb = np.array(resp.embeddings[0].values).reshape(1, -1)
        except Exception as e:
            print(f"Error embedding validation query: {e}")
            continue

        distances, indices = nn_model.kneighbors(val_emb)
        
        similar_examples = []
        for idx in indices[0]:
            similar_examples.append({
                'query': df_train.iloc[idx]['Query'],
                'recall': df_train.iloc[idx]['Recall@10']
            })

        json_response = predict_rank_with_groq(val_query, similar_examples)
        
        final_rank = 0.0
        if json_response:
            try:
                data = json.loads(json_response)
                final_rank = float(data.get("predicted_recall", 0.0))
                print(f"Query: {val_query} | Predicted: {final_rank}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Query: {val_query} | Parse Error: {e}")
                import re
                match = re.search(r"0\.\d+", json_response)
                if match:
                    final_rank = float(match.group())
                    print(f"  -> Recovered via Regex: {final_rank}")
        else:
            print(f"Query: {val_query} | Failed to get response")
        
        results.append(final_rank)
        
        time.sleep(4)

    pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(
    scores, results
    )
    print(pearson_corr, pvaluep, kendall_corr, pvalue)

if __name__ == "__main__":
    main()