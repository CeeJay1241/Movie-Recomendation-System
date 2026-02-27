"""
evaluate.py
Evaluation metrics for the recommendation system.

All metrics are computed with plain NumPy/Pandas — no Surprise dependency.

Implements:
  - RMSE  (root mean squared error)
  - MAE   (mean absolute error)
  - Precision@K — fraction of top-K recommendations the user actually liked
  - Recall@K    — fraction of liked items captured in the top-K list

A rating is considered "liked" if it is at or above `threshold` (default 3.5).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.model import SVDRecommender


def rmse_mae(model: SVDRecommender, test_df: pd.DataFrame) -> dict:
    """
    Compute RMSE and MAE over all (user, movie, true_rating) rows in test_df.

    Rows where the user or movie was not seen during training are skipped
    (cold-start entries cannot be fairly evaluated).
    """
    y_true, y_pred = [], []
    for row in test_df.itertuples(index=False):
        est = model.predict(row.userId, row.movieId)
        y_true.append(row.rating)
        y_pred.append(est)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


def precision_recall_at_k(
    model: SVDRecommender,
    test_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 3.5,
) -> dict:
    """
    Compute Precision@K and Recall@K averaged over all users in the test set.

    Algorithm:
        1. For each user, collect (predicted_rating, true_rating) pairs from test_df.
        2. Sort by predicted rating descending and take the top-K.
        3. Precision@K = |{top-K} ∩ {liked}| / K
        4. Recall@K    = |{top-K} ∩ {liked}| / |{liked in test set}|

    Args:
        model:     Trained SVDRecommender.
        test_df:   DataFrame with columns [userId, movieId, rating].
        k:         Rank cut-off.
        threshold: Minimum true rating to count as "liked".
    """
    # Build per-user list of (predicted_rating, true_rating)
    user_preds = defaultdict(list)
    for row in test_df.itertuples(index=False):
        est = model.predict(row.userId, row.movieId)
        user_preds[row.userId].append((est, row.rating))

    precisions, recalls = [], []
    for uid, pairs in user_preds.items():
        pairs.sort(key=lambda x: x[0], reverse=True)   # sort by predicted ↓
        top_k = pairs[:k]

        n_liked_in_topk = sum(1 for (_, true_r) in top_k if true_r >= threshold)
        n_liked_total = sum(1 for (_, true_r) in pairs if true_r >= threshold)

        precisions.append(n_liked_in_topk / k if k > 0 else 0.0)
        recalls.append(
            n_liked_in_topk / n_liked_total if n_liked_total > 0 else 0.0
        )

    avg_precision = round(float(np.mean(precisions)), 4) if precisions else 0.0
    avg_recall = round(float(np.mean(recalls)), 4) if recalls else 0.0

    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "k": k,
        "threshold": threshold,
    }


def evaluate_model(
    model: SVDRecommender,
    test_df: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Run the full evaluation suite and return a combined results dict."""
    error_metrics = rmse_mae(model, test_df)
    ranking_metrics = precision_recall_at_k(model, test_df, k=k)
    return {**error_metrics, **ranking_metrics}


def print_evaluation_report(metrics: dict) -> None:
    print("\n" + "=" * 40)
    print("  MODEL EVALUATION REPORT")
    print("=" * 40)
    print(f"  RMSE              : {metrics['rmse']}")
    print(f"  MAE               : {metrics['mae']}")
    print(f"  Precision@{metrics['k']:<2}      : {metrics['precision_at_k']}")
    print(f"  Recall@{metrics['k']:<2}         : {metrics['recall_at_k']}")
    print(f"  (liked threshold  : >= {metrics['threshold']} stars)")
    print("=" * 40 + "\n")
