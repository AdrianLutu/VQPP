import pickle
import numpy as np
import pandas as pd

def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]


def compute_metric_of_each_query(group):

    r10 = 0

    for i in range(min(10, len(group))):
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

with open(r".\clip_results\VAST\msrvtt_test_predictions.pickle", "rb") as f:
    predictions = pickle.load(f)

query_results = split_into_groups(predictions, 25)
print(len(query_results))
metrics = [compute_metric_of_each_query(group) for group in query_results]
predicted_r10, predicted_mrr = collect_metrics(metrics)

with open(r".\clip_results\VAST\msrvtt_test_r10.pickle", "wb") as r10:
    pickle.dump(predicted_r10, r10)

with open(r".\clip_results\VAST\msrvtt_test_rr.pickle", "wb") as rr:
    pickle.dump(predicted_mrr, rr)