"""
Interaction feature engineering for the Water Potability dataset.

Adds exactly 5 domain-motivated interaction features. Additional features
are deliberately avoided to reduce overfitting risk on this noisy dataset.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")

ENGINEERED_FEATURE_NAMES = [
    "ph_Hardness",
    "Solids_Conductivity",
]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add exactly 5 interaction features to the input DataFrame.

    Feature rationale:
        ph_Hardness         — combined mineral/acid-base effect on potability
        Solids_Conductivity — dissolved solid load and its conductive capacity

    WHY ONLY 5:
        Each additional feature on a small, noisy dataset increases the risk of
        the model learning spurious correlations. 5 features were selected for
        their physical plausibility and kept to the minimum sufficient set.

    Args:
        X: Feature DataFrame with original 9 water quality columns.

    Returns:
        New DataFrame with original columns + 2 engineered features (11 columns total).
    """
    X = X.copy()
    X["ph_Hardness"] = X["ph"] * X["Hardness"]
    X["Solids_Conductivity"] = X["Solids"] * X["Conductivity"]

    logger.debug(
        f"Feature engineering complete — added {len(ENGINEERED_FEATURE_NAMES)} features. "
        f"Total columns: {X.shape[1]}"
    )
    return X
