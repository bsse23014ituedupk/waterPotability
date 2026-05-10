"""
Feature selection using SelectFromModel with a lightweight Random Forest.

Selects features whose importance is above the median importance of all features
(including engineered ones). Must be fitted exclusively on training data.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def build_selector() -> SelectFromModel:
    """
    Construct a SelectFromModel feature selector.

    Uses a lightweight Random Forest (50 estimators) as the base estimator.
    A low estimator count is intentional: this RF is used only for feature
    ranking, not for final prediction. Keeping it small reduces overfitting
    of the importance scores on the training set.

    Returns:
        Unfitted SelectFromModel instance with threshold='median'.
    """
    base_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    return SelectFromModel(estimator=base_rf, threshold="median")


def fit_selector(
    selector: SelectFromModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> SelectFromModel:
    """
    Fit the feature selector on training data ONLY.

    CRITICAL: Must only be called with (X_train, y_train). Using validation
    or test data here leaks label information into the feature selection step,
    invalidating the model's out-of-sample performance estimates.

    After fitting, logs the selected feature names and their importance scores.

    Args:
        selector: Unfitted SelectFromModel instance.
        X_train:  Training features (post-imputation, post-engineering, post-scaling).
        y_train:  Training labels.

    Returns:
        Fitted SelectFromModel.
    """
    selector.fit(X_train, y_train)

    selected_mask = selector.get_support()
    importances = selector.estimator_.feature_importances_
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [
        f"feature_{i}" for i in range(X_train.shape[1])
    ]

    selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
    rejected_features = [f for f, m in zip(feature_names, selected_mask) if not m]

    logger.info(
        f"Feature selection complete — {len(selected_features)} selected, "
        f"{len(rejected_features)} rejected (threshold='median')"
    )
    logger.info("Selected features and importances:")
    for feat, imp, mask in sorted(
        zip(feature_names, importances, selected_mask),
        key=lambda x: -x[1],
    ):
        status = "✓" if mask else "✗"
        logger.info(f"  [{status}] {feat}: {imp:.4f}")

    return selector


def get_selected_feature_names(
    selector: SelectFromModel, original_feature_names: List[str]
) -> List[str]:
    """
    Retrieve the list of feature names that passed the selection threshold.

    Args:
        selector:               Fitted SelectFromModel.
        original_feature_names: Column names before selection.

    Returns:
        List of selected feature name strings.
    """
    mask = selector.get_support()
    return [name for name, selected in zip(original_feature_names, mask) if selected]
