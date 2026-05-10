"""
Artifact manager — save and load pipeline, model, threshold, and metadata artifacts.
All artifacts are stored under artifacts/models/ by default.
"""

import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def save_artifacts(
    preprocessing_pipeline: Any,
    model: Any,
    optimal_threshold: float,
    config: SimpleNamespace,
) -> None:
    """
    Persist all model artifacts to disk.

    Saves:
        - preprocessing_pipeline.pkl  — fitted sklearn Pipeline
        - xgboost_model.pkl           — trained XGBoost model
        - threshold.json              — optimal decision threshold
        - metadata.json               — version, timestamp, config snapshot

    Args:
        preprocessing_pipeline: Fitted preprocessing Pipeline object.
        model:                  Trained XGBoost classifier.
        optimal_threshold:      Decision threshold selected on validation set.
        config:                 Project configuration namespace.
    """
    artifacts_dir: str = config.api.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save pipeline
    pipeline_path = os.path.join(artifacts_dir, "preprocessing_pipeline.pkl")
    joblib.dump(preprocessing_pipeline, pipeline_path)
    logger.info(f"Preprocessing pipeline saved → {pipeline_path}")

    # Save model
    model_path = os.path.join(artifacts_dir, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"XGBoost model saved → {model_path}")

    # Save threshold
    threshold_path = os.path.join(artifacts_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
    logger.info(f"Threshold saved → {threshold_path} (value={optimal_threshold:.4f})")

    # Save metadata
    metadata = {
        "version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "optimal_threshold": optimal_threshold,
        "config": {
            "training_random_state": config.training.random_state,
        },
    }
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved → {metadata_path}")


def load_artifact(path: str) -> Any:
    """
    Load a joblib-serialized artifact from disk.

    Args:
        path: Absolute or relative path to .pkl file.

    Returns:
        Deserialized Python object.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Save the preprocessed and split data to CSV for auditability.
    Files are saved in artifacts/data/.

    Args:
        X_train, y_train: Preprocessed training data.
        X_val, y_val:     Preprocessed validation data.
        X_test, y_test:   Preprocessed test data.
    """
    data_dir = "artifacts/data"
    os.makedirs(data_dir, exist_ok=True)

    # Combine X and y for each split
    splits = [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]

    for name, X, y in splits:
        df = pd.concat([X, y], axis=1)
        path = os.path.join(data_dir, f"{name}_processed.csv")
        df.to_csv(path, index=False)
        logger.info(f"Processed {name} data saved → {path}")
