import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kendalltau

with open(r".\clip_results\VAST\msrvtt_train_rr.pickle", "rb") as ct:
    clip_train = pickle.load(ct)

with open(r".\clip_results\VAST\msrvtt_val_rr.pickle", "rb") as cv:
    clip_val = pickle.load(cv)

with open(r".\clip_results\VAST\msrvtt_test_rr.pickle", "rb") as ctst:
    clip_test = pickle.load(ctst)

bert_train_df = pd.read_csv(r".\finetune_bert_result\VAST\RR\msrvtt_train_predictions.csv")
bert_val_df = pd.read_csv(r".\finetune_bert_result\VAST\RR\msrvtt_val_predictions.csv")
bert_test_df = pd.read_csv(r".\finetune_bert_result\VAST\RR\msrvtt_test_predictions.csv")

bert_train = bert_train_df['Predicted_Score'].to_list()
bert_val = bert_val_df['Predicted_Score'].to_list()
bert_test = bert_test_df['Predicted_Score'].to_list()

def pair_estimators(bert_scores, clip_scores, split):
    """Pairs the BERT score and the CLIP score of the same query.

    The pairing is positional, so it is only correct when both lists cover the same
    queries in the same order: the BERT scores come from the rows of the predictions
    CSV, the CLIP scores from get_metrics.py. zip() stops at the shorter list without
    reporting anything, which pairs scores belonging to different queries and only
    surfaces later as an unrelated error, so the lengths are checked here instead.
    """
    if len(bert_scores) != len(clip_scores):
        raise ValueError(
            f"{split}: {len(bert_scores)} BERT scores and {len(clip_scores)} CLIP "
            "scores. The two estimators must cover the same queries, in the same "
            "order; regenerate them before training the meta model."
        )
    return list(zip(bert_scores, clip_scores))


train = pair_estimators(bert_train, clip_train, "train")
val = pair_estimators(bert_val, clip_val, "val")
test = pair_estimators(bert_test, clip_test, "test")


train_target = bert_train_df['Reciprocal_Rank'].to_list()
val_target = bert_val_df['Reciprocal_Rank'].to_list()
test_target = bert_test_df['Reciprocal_Rank'].to_list()


kernel = ["linear", "poly", "rbf"]
C = [1.0, 0.1, 0.01]
best_mse = 99999999
best_k = None
best_c = None

for k in kernel:
    for c in C:
        model = SVR(kernel=k, C=c)

        model.fit(train, train_target)

        val_preds = model.predict(val)

        mse = mean_squared_error(val_target, val_preds)
        print(k, c, mse)

        if mse < best_mse:
            best_mse = mse
            best_k = k
            best_c = c

best_model = SVR(kernel=best_k, C=best_c)
best_model.fit(train, train_target)
test_preds = best_model.predict(test)

pearson, p_value_person = pearsonr(test_preds, test_target)
kendall, p_value_kendall = kendalltau(test_preds, test_target)

print(pearson, p_value_person)
print(kendall, p_value_kendall)
