import pickle
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, kendalltau

with open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_results.pickle", "rb") as f:
    predictions = pickle.load(f)
print(f"Length {len(predictions)}")
df = pd.read_csv("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/metrics/msrvtt_test.csv")
rr = df["Reciprocal_Rank"].tolist()
r10 = df["Recall@10"].tolist()

def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]


def compute_metric_of_each_query(group):

    r10 = 0

    for i in range(10):
        if group[i] >= 0.5:
            r10 += 1
    r10 /= max(1, sum(1 for score in group if score >= 0.5))

    mrr = 0

    for i in range(len(group)):
        if group[i] >= 0.5:
            mrr = 1 / (i + 1)
            break


    return (r10, mrr)


def collect_metrics(all_metrics):
    r10s = [metric[0] for metric in all_metrics]
    mrrs = [metric[1] for metric in all_metrics]
    return r10s, mrrs


def compute_correlations(map1, map2, title):

    print(len(map1))
    print(len(map2))

    (pearson, p_value_pearson) = pearsonr(map1, map2)
    (kendall, p_value_kendall) = kendalltau(map1, map2)

    print(title)
    print("Pearson Correlation {} p-value {} ".format(pearson, p_value_pearson))
    print("Kendall Correlation {} p-value {} ".format(kendall, p_value_kendall))
    print()


query_results = split_into_groups(predictions, 25)
print(len(query_results))
metrics = [compute_metric_of_each_query(group) for group in query_results]
predicted_r10, predicted_mrr = collect_metrics(metrics)


compute_correlations(predicted_r10, r10, "  R10 Correlations")
compute_correlations(predicted_mrr, rr, "RR Correlations")
