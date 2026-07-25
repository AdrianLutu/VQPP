import pickle

# The CLIP classifier stores probabilities, so a candidate counts as predicted
# relevant when its probability is at least 0.5. Keep this file in sync with
# baselines/finetune_clip/compute_correlations.py, which computes the same metrics.
POSITIVE_THRESHOLD = 0.5

# Every split needs the predictions and the dataset they were produced from: the
# dataset holds the query index of each candidate, which is what defines the groups.
# meta_svr.py consumes all three splits, so all three are written here.
SPLITS = {
    "train": {
        "predictions": r".\clip_results\VAST\msrvtt_train_predictions.pickle",
        "dataset": r".\clip_datasets\VAST\msrvtt_clip_train.pickle",
        "r10": r".\clip_results\VAST\msrvtt_train_r10.pickle",
        "rr": r".\clip_results\VAST\msrvtt_train_rr.pickle",
        "query_indices": r".\clip_results\VAST\msrvtt_train_query_indices.pickle",
    },
    "val": {
        "predictions": r".\clip_results\VAST\msrvtt_val_predictions.pickle",
        "dataset": r".\clip_datasets\VAST\msrvtt_clip_val.pickle",
        "r10": r".\clip_results\VAST\msrvtt_val_r10.pickle",
        "rr": r".\clip_results\VAST\msrvtt_val_rr.pickle",
        "query_indices": r".\clip_results\VAST\msrvtt_val_query_indices.pickle",
    },
    "test": {
        "predictions": r".\clip_results\VAST\msrvtt_test_predictions.pickle",
        "dataset": r".\clip_datasets\VAST\msrvtt_clip_test.pickle",
        "r10": r".\clip_results\VAST\msrvtt_test_r10.pickle",
        "rr": r".\clip_results\VAST\msrvtt_test_rr.pickle",
        "query_indices": r".\clip_results\VAST\msrvtt_test_query_indices.pickle",
    },
}


def group_predictions_by_query(predictions, dataset, split):
    """Groups the flat prediction list by the query index stored in the dataset.

    The predictions are produced by a DataLoader built with shuffle=False, so element
    i of the list corresponds to sample i of the dataset.
    """
    if not dataset:
        raise ValueError(f"The {split} dataset is empty.")

    if len(predictions) != len(dataset):
        raise ValueError(
            f"{split}: {len(predictions)} predictions for {len(dataset)} samples: the "
            "predictions and the dataset do not come from the same run."
        )

    if len(dataset[0]) < 3:
        raise ValueError(
            f"The {split} dataset uses the old (features, label) format. Regenerate it "
            "with create_dataset_clip.py so every sample carries its query index."
        )

    groups = {}
    for prediction, sample in zip(predictions, dataset):
        groups.setdefault(sample[2], []).append(prediction)
    return groups


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


def process_split(split, paths):
    with open(paths["predictions"], "rb") as f:
        predictions = pickle.load(f)

    with open(paths["dataset"], "rb") as f:
        dataset = pickle.load(f)

    query_results = group_predictions_by_query(predictions, dataset, split)
    # The query indices are written next to the metrics so that meta_svr.py can line
    # these values up with the BERT predictions instead of assuming both lists cover
    # the same queries in the same order.
    query_indices = sorted(query_results)
    print(f"{split}: {len(query_indices)} queries")

    metrics = [compute_metric_of_each_query(query_results[index]) for index in query_indices]
    predicted_r10, predicted_rr = collect_metrics(metrics)

    with open(paths["r10"], "wb") as f:
        pickle.dump(predicted_r10, f)

    with open(paths["rr"], "wb") as f:
        pickle.dump(predicted_rr, f)

    with open(paths["query_indices"], "wb") as f:
        pickle.dump(query_indices, f)


if __name__ == "__main__":
    for split, paths in SPLITS.items():
        process_split(split, paths)