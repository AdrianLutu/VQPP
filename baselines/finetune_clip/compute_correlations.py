import math
import pickle
import pandas as pd

from scipy.stats import pearsonr, kendalltau

PREDICTIONS_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_results.pickle"
TEST_DATASET_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_clip_test.pickle"
METRICS_CSV_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/metrics/msrvtt_test.csv"

# finetune_clip.py stores probabilities, so a candidate counts as predicted relevant
# when its probability is at least 0.5.
POSITIVE_THRESHOLD = 0.5

# Number of candidates per query assumed by datasets built before the query index was
# stored with every sample.
FALLBACK_GROUP_SIZE = 25


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def as_probabilities(predictions):
    """Converts predictions saved before finetune_clip.py applied the sigmoid itself.

    Those runs stored raw logits, which would make POSITIVE_THRESHOLD mean a
    probability of 0.62 instead of 0.5.
    """
    if all(0.0 <= prediction <= 1.0 for prediction in predictions):
        return predictions

    print("Predictions fall outside [0, 1], so they are logits: applying a sigmoid.")
    return [sigmoid(prediction) for prediction in predictions]


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

    if len(dataset[0]) < 3:
        # Datasets built before the query index was stored still load, using the old
        # fixed-size grouping.
        print(
            f"The dataset carries no query index, falling back to groups of "
            f"{FALLBACK_GROUP_SIZE}. Regenerate it with create_dataset_clip.py: a query "
            "that yields fewer candidates shifts every query after it."
        )
        return {
            index: predictions[start : start + FALLBACK_GROUP_SIZE]
            for index, start in enumerate(range(0, len(predictions), FALLBACK_GROUP_SIZE))
        }

    groups = {}
    for prediction, sample in zip(predictions, dataset):
        groups.setdefault(sample[2], []).append(prediction)
    return groups


def check_alignment(dataset, df):
    """Checks that a query index points at the CSV row of the same query."""
    if len(dataset[0]) < 4 or "Query" not in df.columns:
        print("No query text in the dataset or no Query column, skipping the check.")
        return

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
predictions = as_probabilities(predictions)

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
