"""
Overfitting detection and gap analysis for the Water Potability model.

Computes train-validation performance gaps and emits structured warnings
when overfitting or model degradation conditions are detected.

Warning thresholds (per Section 10 of the master prompt):
    Train-Val accuracy gap > 0.10  → OVERFITTING WARNING
    Train-Val F1 gap       > 0.10  → OVERFITTING WARNING
    Train-Val AUC gap      > 0.10  → OVERFITTING WARNING
    Val accuracy < 0.55            → MODEL UNDERPERFORMING
    Val precision < 0.50           → PRECISION COLLAPSE
    Val recall > 0.90              → RECALL INFLATION (threshold too low)
"""

from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def check_overfitting(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute train-validation performance gaps and emit targeted warnings.

    Analyses three categories of problems:
        1. OVERFITTING — large train-val gaps indicate memorisation
        2. UNDERPERFORMANCE — absolute validation scores below acceptable floor
        3. METRIC IMBALANCE — precision collapse or recall inflation

    Args:
        train_metrics: Metrics dict for the training split.
        val_metrics:   Metrics dict for the validation split.
        test_metrics:  Metrics dict for the test split.

    Returns:
        Dict with keys: accuracy_gap, f1_gap, auc_gap, val_test_accuracy_delta.
    """
    acc_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    f1_gap  = train_metrics["f1"]       - val_metrics["f1"]
    auc_gap = train_metrics["roc_auc"]  - val_metrics["roc_auc"]

    val_test_acc_delta = abs(val_metrics["accuracy"] - test_metrics["accuracy"])

    logger.info("=" * 50)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 50)
    logger.info(f"  Train-Val Accuracy Gap: {acc_gap:+.4f}")
    logger.info(f"  Train-Val F1 Gap:       {f1_gap:+.4f}")
    logger.info(f"  Train-Val AUC Gap:      {auc_gap:+.4f}")
    logger.info(f"  Val-Test Accuracy Δ:    {val_test_acc_delta:.4f}")

    # Overfitting warnings
    if acc_gap > 0.10:
        logger.warning(
            f"⚠️  OVERFITTING DETECTED — Train-Val Accuracy Gap: {acc_gap:.3f} > 0.10"
        )
    else:
        logger.info(f"✓ Accuracy gap OK ({acc_gap:.3f} ≤ 0.10)")

    if f1_gap > 0.10:
        logger.warning(
            f"⚠️  OVERFITTING DETECTED — Train-Val F1 Gap: {f1_gap:.3f} > 0.10"
        )
    else:
        logger.info(f"✓ F1 gap OK ({f1_gap:.3f} ≤ 0.10)")

    if auc_gap > 0.10:
        logger.warning(
            f"⚠️  OVERFITTING DETECTED — Train-Val AUC Gap: {auc_gap:.3f} > 0.10"
        )
    else:
        logger.info(f"✓ AUC gap OK ({auc_gap:.3f} ≤ 0.10)")

    # Absolute performance floors
    if val_metrics["accuracy"] < 0.55:
        logger.warning(
            f"⚠️  MODEL UNDERPERFORMING — Val Accuracy: {val_metrics['accuracy']:.3f} < 0.55"
        )
    if val_metrics["precision"] < 0.50:
        logger.warning(
            f"⚠️  PRECISION COLLAPSE — Val Precision: {val_metrics['precision']:.3f} < 0.50"
        )
    if val_metrics["recall"] > 0.90:
        logger.warning(
            f"⚠️  RECALL INFLATION — Val Recall: {val_metrics['recall']:.3f} > 0.90 "
            f"(threshold may be too low)"
        )

    # Val-Test consistency
    if val_test_acc_delta > 0.05:
        logger.warning(
            f"⚠️  VAL-TEST INCONSISTENCY — Accuracy delta: {val_test_acc_delta:.3f} > 0.05"
        )
    else:
        logger.info(f"✓ Val-Test accuracy delta OK ({val_test_acc_delta:.3f} ≤ 0.05)")

    logger.info("=" * 50)

    return {
        "accuracy_gap":         acc_gap,
        "f1_gap":               f1_gap,
        "auc_gap":              auc_gap,
        "val_test_acc_delta":   val_test_acc_delta,
    }
