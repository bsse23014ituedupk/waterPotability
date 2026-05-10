"""
Schema and distribution validation for the Water Potability dataset.

Provides post-split distribution checks to verify that class ratios
are maintained across train, validation, and test splits.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def validate_split_distributions(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Validate that class distributions are consistent across all three splits.

    Computes the minority class ratio for each split and warns if any split
    deviates more than ``tolerance`` from the overall dataset ratio.

    Args:
        y_train:   Training target labels.
        y_val:     Validation target labels.
        y_test:    Test target labels.
        tolerance: Maximum allowed deviation from the mean class ratio.

    Returns:
        Dict with keys 'train_ratio', 'val_ratio', 'test_ratio'.
    """
    train_ratio = float(y_train.mean())
    val_ratio = float(y_val.mean())
    test_ratio = float(y_test.mean())
    overall_ratio = float(pd.concat([y_train, y_val, y_test]).mean())

    logger.info("Validating class distribution across splits:")
    logger.info(f"  Overall positive class ratio: {overall_ratio:.4f}")
    logger.info(f"  Train ratio:      {train_ratio:.4f}")
    logger.info(f"  Validation ratio: {val_ratio:.4f}")
    logger.info(f"  Test ratio:       {test_ratio:.4f}")

    for split_name, ratio in [
        ("train", train_ratio),
        ("validation", val_ratio),
        ("test", test_ratio),
    ]:
        deviation = abs(ratio - overall_ratio)
        if deviation > tolerance:
            logger.warning(
                f"⚠️  Class imbalance deviation in {split_name} split: "
                f"{deviation:.4f} > tolerance {tolerance:.4f}"
            )
        else:
            logger.debug(f"  {split_name} distribution OK (deviation={deviation:.4f})")

    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
