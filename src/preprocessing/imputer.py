"""
Median imputation transformer (fit-on-train-only).

Wraps sklearn's SimpleImputer and logs the learned median values per feature
after fitting, providing full observability of imputation behaviour.
"""

import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def build_imputer() -> SimpleImputer:
    """
    Construct a median imputer.

    Median is chosen over mean because:
    - This dataset contains extreme outliers (Solids, Conductivity).
    - Median is resistant to skew and outliers; mean would shift fill values.

    Returns:
        Unfitted SimpleImputer instance with strategy='median'.
    """
    return SimpleImputer(strategy="median")


def fit_imputer(imputer: SimpleImputer, X_train: pd.DataFrame) -> SimpleImputer:
    """
    Fit the imputer on training data ONLY and log learned median values.

    CRITICAL: This function must only ever be called with X_train.
    Calling it with X_val or X_test would constitute data leakage.

    Args:
        imputer: Unfitted SimpleImputer instance.
        X_train: Training feature DataFrame.

    Returns:
        Fitted SimpleImputer.
    """
    imputer.fit(X_train)
    logger.info("Imputer fitted on X_train. Learned medians:")
    for col, median_val in zip(X_train.columns, imputer.statistics_):
        logger.info(f"  {col}: {median_val:.4f}")
    return imputer
