"""
Dataset loading and schema validation for the Water Potability Prediction System.

Validates the CSV schema, logs class distribution and missing-value counts,
and raises informative errors if critical columns are absent.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")

EXPECTED_FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"
ALL_EXPECTED_COLUMNS = EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]


def load_and_validate(filepath: str) -> pd.DataFrame:
    """
    Load the water potability CSV, validate schema, and log diagnostics.

    Operations performed:
        1. Read CSV from ``filepath``.
        2. Verify all expected columns are present (raises ValueError if not).
        3. Log class distribution (counts and percentages).
        4. Log missing-value counts per feature column.
        5. Return the raw DataFrame with the target column intact.

    Args:
        filepath: Path to water_potability.csv.

    Returns:
        Raw pd.DataFrame with all original columns preserved.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If one or more required columns are missing.
    """
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded — shape: {df.shape}")

    # Schema validation
    missing_cols = [c for c in ALL_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in dataset: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info("Schema validation passed — all expected columns present.")

    # Class distribution
    class_counts = df[TARGET_COLUMN].value_counts()
    class_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    logger.info("Class distribution:")
    for cls in sorted(class_counts.index):
        label = "Potable" if cls == 1 else "Not Potable"
        logger.info(f"  Class {cls} ({label}): {class_counts[cls]} samples ({class_pct[cls]:.1f}%)")

    # Missing value counts
    logger.info("Missing values per feature:")
    for col in EXPECTED_FEATURE_COLUMNS:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        if n_missing > 0:
            logger.info(f"  {col}: {n_missing} missing ({pct_missing:.1f}%)")
        else:
            logger.debug(f"  {col}: 0 missing")

    total_missing = df[EXPECTED_FEATURE_COLUMNS].isna().sum().sum()
    logger.info(f"Total missing values across all features: {total_missing}")

    return df
