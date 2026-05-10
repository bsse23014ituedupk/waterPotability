"""
Full preprocessing pipeline assembly for the Water Potability Prediction System.

Assembles imputation → feature engineering → scaling → feature selection into
a single serialisable sklearn Pipeline. Strictly follows the mandated data-split
order to prevent any form of data leakage.

MANDATORY ORDER (per Section 3 of the master prompt):
    STEP 4: Fit imputer on X_train only → transform all splits
    STEP 5: Apply feature engineering (no fitting required — pure transforms)
    STEP 6: Fit RobustScaler on X_train only → transform all splits
    STEP 7: Fit feature selector on X_train only → transform all splits
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing.imputer import build_imputer, fit_imputer
from src.preprocessing.feature_engineer import engineer_features, ENGINEERED_FEATURE_NAMES
from src.preprocessing.scaler import build_scaler, fit_scaler
from src.preprocessing.feature_selector import (
    build_selector,
    fit_selector,
    get_selected_feature_names,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


# ---------------------------------------------------------------------------
# Custom sklearn-compatible transformers for Pipeline compatibility
# ---------------------------------------------------------------------------

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around engineer_features.
    Stateless: no fitting required, always safe to apply to any split.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            raise ValueError("FeatureEngineerTransformer requires a pandas DataFrame.")
        return engineer_features(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array(list(input_features) + ENGINEERED_FEATURE_NAMES)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn imputer wrapper that preserves DataFrame column names throughout
    the pipeline (SimpleImputer returns numpy arrays by default).
    """

    def __init__(self):
        from sklearn.impute import SimpleImputer
        self._imputer = SimpleImputer(strategy="median")
        self._columns = None

    def fit(self, X, y=None):
        self._columns = list(X.columns)
        self._imputer.fit(X)
        logger.info("DataFrameImputer fitted — medians learned:")
        for col, median_val in zip(self._columns, self._imputer.statistics_):
            logger.info(f"  {col}: {median_val:.4f}")
        return self

    def transform(self, X):
        arr = self._imputer.transform(X)
        return pd.DataFrame(arr, columns=self._columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._columns) if self._columns else np.array([])


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    RobustScaler wrapper that preserves DataFrame column names.
    """

    def __init__(self):
        from sklearn.preprocessing import RobustScaler
        self._scaler = RobustScaler()
        self._columns = None

    def fit(self, X, y=None):
        self._columns = list(X.columns)
        self._scaler.fit(X)
        logger.info("DataFrameScaler (RobustScaler) fitted on training data.")
        return self

    def transform(self, X):
        arr = self._scaler.transform(X)
        return pd.DataFrame(arr, columns=self._columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._columns) if self._columns else np.array([])


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    SelectFromModel wrapper that preserves feature names after selection.
    """

    def __init__(self):
        self._selector = build_selector()
        self._selected_columns = None

    def fit(self, X, y=None):
        self._selector = fit_selector(self._selector, X, y)
        mask = self._selector.get_support()
        cols = list(X.columns) if hasattr(X, "columns") else [
            f"f{i}" for i in range(X.shape[1])
        ]
        self._selected_columns = [c for c, m in zip(cols, mask) if m]
        return self

    def transform(self, X):
        arr = self._selector.transform(X)
        index = X.index if hasattr(X, "index") else None
        return pd.DataFrame(arr, columns=self._selected_columns, index=index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._selected_columns) if self._selected_columns else np.array([])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct the full preprocessing Pipeline.

    Pipeline steps:
        1. imputer   — median imputation (preserves DataFrame)
        2. engineer  — interaction feature engineering (stateless)
        3. scaler    — RobustScaler (preserves DataFrame)

    Returns:
        Unfitted sklearn Pipeline.
    """
    pipeline = Pipeline(steps=[
        ("imputer",  DataFrameImputer()),
        ("engineer", FeatureEngineerTransformer()),
        ("scaler",   DataFrameScaler()),
    ])
    logger.info("Preprocessing pipeline constructed with 3 steps: imputer → engineer → scaler")
    return pipeline


def get_pipeline_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Retrieve the final feature names after all pipeline transformations.

    Args:
        pipeline: Fitted preprocessing Pipeline.

    Returns:
        List of feature name strings after selection.
    """
    scaler_step: DataFrameScaler = pipeline.named_steps["scaler"]
    return list(scaler_step.get_feature_names_out())
