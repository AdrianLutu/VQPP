import pickle
import pandas as pd

from scipy.stats import pearsonr, kendalltau

PREDICTIONS_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_results.pickle"
TEST_DATASET_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_clip_test.pickle"
METRICS_CSV_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/metrics/msrvtt_test.csv"

# finetune_clip.py stores probabilities, so a candidate counts as predicted relevant
# when its probability is at least 0.5.
POSITIVE_THRESHOLD = 0.5


def group_predictions_by_query(predictions, dataset):
    """Groups the flat prediction list by the query index stored in the dataset.

    The predictions are produced by a DataLoader built with shuffle=False, so element
    i of the list corresponds to sample i of the dataset.
    """
    if not dataset:
        raise ValueError(f"The dataset at {TEST_DATASET_PATH} is empty.")

    if len(predictions) != len(dataset):
        raise ValueError(
            f"{len(predictions)} predictions for {len(dataset)} samples: the "
            "predictions and the dataset do not come from the same run."
        )

    if len(dataset[0]) < 4:
        raise ValueError(
            "The dataset uses the old (features, label) format. Regenerate it with "
            "create_dataset_clip.py so every sample carries its query index and text."
        )

    groups = {}
    for prediction, sample in zip(predictions, dataset):
        groups.setdefault(sample[2], []).append(prediction)
    return groups


def check_alignment(dataset, df):
    """Checks that a query index points at the CSV row of the same query."""
    for sample in dataset:
        csv_query = str(df["Query"].iloc[sample[2]]).strip()
        if csv_query != str(sample[3]).strip():
            raise ValueError(
                f"Query {sample[2]} is '{sample[3]}' in the jsonl but '{csv_query}' in "
                "the metrics CSV: the two files are not in the same order."
            )


def compute_metric_of_each_query(group):
    """Predicted Recall@10 and Reciprocal Rank for a single query.

    group holds the predicted probability that each candidate is the ground-truth
    video, in the order the retrieval system ranked the candidates.
    """
    # Recall@10: relevant candidates retrieved in the top 10, over the total number of
    # relevant candidates. The relevant set is estimated by the candidates the model
    # predicts as relevant. Slicing the group keeps this correct for queries with
    # fewer than 10 candidates.
    predicted_relevant = [score >= POSITIVE_THRESHOLD for score in group]
    total_relevant = sum(predicted_relevant)
    r10 = sum(predicted_relevant[:10]) / total_relevant if total_relevant else 0.0

    # Reciprocal rank: the inverse of the rank of the first relevant candidate, 0 when
    # no candidate is predicted relevant.
    rr = 0.0
    for rank, score in enumerate(group, start=1):
        if score >= POSITIVE_THRESHOLD:
            rr = 1.0 / rank
            break

    return (r10, rr)


def collect_metrics(all_metrics):
    r10s = [metric[0] for metric in all_metrics]
    rrs = [metric[1] for metric in all_metrics]
    return r10s, rrs


def compute_correlations(map1, map2, title):

    print(len(map1))
    print(len(map2))

    (pearson, p_value_pearson) = pearsonr(map1, map2)
    (kendall, p_value_kendall) = kendalltau(map1, map2)

    print(title)
    print("Pearson Correlation {} p-value {} ".format(pearson, p_value_pearson))
    print("Kendall Correlation {} p-value {} ".format(kendall, p_value_kendall))
    print()


with open(PREDICTIONS_PATH, "rb") as f:
    predictions = pickle.load(f)
print(f"Length {len(predictions)}")

with open(TEST_DATASET_PATH, "rb") as f:
    test_dataset = pickle.load(f)

df = pd.read_csv(METRICS_CSV_PATH)

check_alignment(test_dataset, df)
query_results = group_predictions_by_query(predictions, test_dataset)
query_indices = sorted(query_results)
print(f"{len(query_indices)} queries evaluated out of {len(df)} rows in the metrics CSV")

metrics = [compute_metric_of_each_query(query_results[index]) for index in query_indices]
predicted_r10, predicted_mrr = collect_metrics(metrics)

# Compare only against the queries that actually produced predictions, otherwise the
# ground truth would be shifted with respect to the predictions.
r10 = df["Recall@10"].iloc[query_indices].tolist()
rr = df["Reciprocal_Rank"].iloc[query_indices].tolist()

compute_correlations(predicted_r10, r10, "  R10 Correlations")
compute_correlations(predicted_mrr, rr, "RR Correlations")
