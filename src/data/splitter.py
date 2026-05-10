"""
Stratified train / validation / test splitting for the Water Potability dataset.

Implements a two-stage stratified split to produce three non-overlapping splits
while preserving the original class ratio in each subset.

Split ratios (default from config):  70% train | 15% validation | 15% test
"""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: SimpleNamespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a stratified two-stage split into train / validation / test subsets.

    Stage 1: Split (X, y) into (train_full, temp) using test_size = val_size + test_size.
    Stage 2: Split (temp) into (validation, test) with equal halves.

    This guarantees class proportions are preserved in all three splits.

    Args:
        X:      Feature DataFrame (before any preprocessing).
        y:      Target Series.
        config: Configuration namespace (uses config.data.*).

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    val_size: float = config.data.val_size
    test_size: float = config.data.test_size
    random_state: int = config.data.random_state

    # Stage 1: Hold out val + test together
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # Stage 2: Split temp into equal val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    # Logging
    logger.info("Stratified split complete:")
    logger.info(f"  Train:      {len(X_train):>5} samples | class ratio: {y_train.mean():.3f}")
    logger.info(f"  Validation: {len(X_val):>5} samples | class ratio: {y_val.mean():.3f}")
    logger.info(f"  Test:       {len(X_test):>5} samples | class ratio: {y_test.mean():.3f}")

    _assert_no_leakage(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _assert_no_leakage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Assert that the three splits share no common index values.

    Raises:
        AssertionError: If any index overlap is detected between splits.
    """
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert len(train_idx & val_idx) == 0, "DATA LEAKAGE: Train and Val share indices!"
    assert len(train_idx & test_idx) == 0, "DATA LEAKAGE: Train and Test share indices!"
    assert len(val_idx & test_idx) == 0, "DATA LEAKAGE: Val and Test share indices!"
    logger.debug("Split index integrity check passed — no overlaps detected.")
