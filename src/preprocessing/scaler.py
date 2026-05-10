"""
RobustScaler wrapper (fit-on-train-only).

RobustScaler is preferred over StandardScaler for this dataset because
it uses IQR and median, making it resistant to the extreme outliers
present in Solids and Conductivity features.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def build_scaler() -> RobustScaler:
    """
    Construct a RobustScaler.

    Why RobustScaler over StandardScaler:
        - StandardScaler uses mean and std — both are sensitive to extreme values.
        - This dataset has confirmed outliers in Solids (max ~60k vs median ~20k)
          and Conductivity.
        - RobustScaler uses Q1, Q3 (IQR) and median → outliers do not distort scaling.

    Returns:
        Unfitted RobustScaler instance.
    """
    return RobustScaler()


def fit_scaler(scaler: RobustScaler, X_train: pd.DataFrame) -> RobustScaler:
    """
    Fit the scaler on training data ONLY and log scale and center statistics.

    CRITICAL: Must only be called with X_train. Fitting on X_val or X_test
    would allow validation/test distribution to influence preprocessing,
    causing data leakage.

    Args:
        scaler:  Unfitted RobustScaler instance.
        X_train: Feature DataFrame after imputation and feature engineering.

    Returns:
        Fitted RobustScaler.
    """
    scaler.fit(X_train)
    logger.info("RobustScaler fitted on X_train.")
    logger.debug("Scaler centers (medians):")
    for col, center in zip(X_train.columns, scaler.center_):
        logger.debug(f"  {col}: center={center:.4f}")
    return scaler
