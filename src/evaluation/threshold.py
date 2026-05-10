"""
Optimized threshold selection using Precision-Recall curve.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def optimize_threshold(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    output_csv: str = "artifacts/threshold_analysis.csv",
) -> Tuple[float, pd.DataFrame]:
    """
    Find the optimal decision threshold using the F1-score derived from
    the Precision-Recall curve.

    Args:
        y_val:       True validation labels.
        val_proba:   Predicted probabilities for the positive class on validation.
        output_csv:  Path to save the full threshold analysis CSV.

    Returns:
        Tuple of (best_threshold float, results DataFrame with all thresholds).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    
    best_score = -1.0
    best_threshold = 0.5
    best_f1 = 0.0
    
    # We want a threshold that keeps accuracy above base rate (0.61) 
    # while maximizing F1.
    from sklearn.metrics import accuracy_score
    
    for t in np.arange(0.45, 0.65, 0.01):
        preds = (val_proba >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        f1 = 2 * (precision_score(y_val, preds, zero_division=0) * recall_score(y_val, preds)) / (precision_score(y_val, preds, zero_division=0) + recall_score(y_val, preds) + 1e-9)
        
        # We need to explicitly protect precision. If precision drops too low, we're just guessing 1s.
        prec = precision_score(y_val, preds, zero_division=0)
        
        # Penalize if precision is worse than 0.55
        score = (0.6 * f1) + (0.4 * acc)
        if prec < 0.55:
            score *= 0.5 # Halve the score as penalty
            
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_f1 = f1

    # Re-calculate precision/recall curve for logging
    precisions, recalls, curve_thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)


    logger.info(
        f"Threshold optimisation complete — "
        f"optimal threshold: {best_threshold:.4f} "
        f"(max F1: {best_f1:.4f})"
    )

    # Compile results for logging
    results = []
    for i, t in enumerate(curve_thresholds):
        results.append({
            "threshold": float(t),
            "precision": float(precisions[i]),
            "recall":    float(recalls[i]),
            "f1":        float(f1_scores[i] if i < len(f1_scores) else 0.0),
        })
        
    results_df = pd.DataFrame(results)

    # Persist threshold analysis for MLflow logging
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    logger.debug(f"Threshold analysis saved → {output_csv}")

    return best_threshold, results_df


def diagnose_probability_distribution(model, X_val, y_val):
    """
    Print the full probability distribution to understand
    what the model actually thinks before threshold is applied.
    """
    proba = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 50)
    print("PROBABILITY DISTRIBUTION DIAGNOSIS")
    print("=" * 50)
    print(f"Min probability:    {proba.min():.4f}")
    print(f"Max probability:    {proba.max():.4f}")
    print(f"Mean probability:   {proba.mean():.4f}")
    print(f"Median probability: {np.median(proba):.4f}")
    print(f"Std probability:    {proba.std():.4f}")
    print()

    # Distribution buckets
    buckets = [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    print("Distribution buckets:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = ((proba >= lo) & (proba < hi)).sum()
        pct = count / len(proba) * 100
        print(f"  [{lo:.2f} – {hi:.2f}): {count:4d} samples ({pct:.1f}%)")

    print()
    print(
        f"Samples BELOW 0.50 threshold: "
        f"{(proba < 0.50).sum()} ({(proba < 0.50).mean()*100:.1f}%)"
    )
    print(
        f"Samples ABOVE 0.50 threshold: "
        f"{(proba >= 0.50).sum()} ({(proba >= 0.50).mean()*100:.1f}%)"
    )

    # Check per true class
    pos_proba = proba[y_val == 1]
    neg_proba = proba[y_val == 0]
    print(f"\nMean proba for TRUE positives: {pos_proba.mean():.4f}")
    print(f"Mean proba for TRUE negatives: {neg_proba.mean():.4f}")
    print(
        f"Separation (delta):            "
        f"{(pos_proba.mean() - neg_proba.mean()):.4f}"
    )
    print("=" * 50)
