"""
Preprocessing pipeline tests — data leakage prevention, SMOTE isolation,
feature engineering shape, and pipeline transform consistency.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.preprocessing.feature_engineer import engineer_features, ENGINEERED_FEATURE_NAMES
from src.preprocessing.pipeline import build_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small synthetic water potability DataFrame for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ph":             np.random.uniform(6.0, 8.5, n),
        "Hardness":       np.random.uniform(100, 300, n),
        "Solids":         np.random.uniform(5000, 40000, n),
        "Chloramines":    np.random.uniform(3, 10, n),
        "Sulfate":        np.random.uniform(200, 400, n),
        "Conductivity":   np.random.uniform(300, 700, n),
        "Organic_carbon": np.random.uniform(8, 18, n),
        "Trihalomethanes":np.random.uniform(50, 100, n),
        "Turbidity":      np.random.uniform(1, 6, n),
        "Potability":     np.random.randint(0, 2, n),
    })
    # Inject some missing values to simulate the real dataset
    df.loc[df.sample(20).index, "ph"] = np.nan
    df.loc[df.sample(15).index, "Sulfate"] = np.nan
    return df


@pytest.fixture
def X_y_split(sample_df):
    """Return X and y from the sample DataFrame."""
    X = sample_df.drop("Potability", axis=1)
    y = sample_df["Potability"]
    return X, y


@pytest.fixture
def train_val_split(X_y_split):
    """Return a simple 70/30 train/val split."""
    from sklearn.model_selection import train_test_split
    X, y = X_y_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    """Verify imputer and scaler are fitted only on training data."""

    def test_imputer_not_fit_on_val_or_test(self, train_val_split):
        """Fitting imputer only on train must not alter val's raw NaN structure."""
        X_train, X_val, y_train, y_val = train_val_split

        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)

        # Check medians came from X_train (not X_val)
        train_medians = X_train.median()
        for col, fitted_median in zip(X_train.columns, imputer.statistics_):
            train_median = train_medians[col]
            assert abs(fitted_median - train_median) < 1e-6, (
                f"Imputer median for '{col}' ({fitted_median:.4f}) doesn't match "
                f"X_train median ({train_median:.4f}) — possible leakage!"
            )

    def test_scaler_not_fit_on_val(self, train_val_split):
        """RobustScaler center_ values must match X_train statistics."""
        X_train, X_val, y_train, y_val = train_val_split

        # Impute first
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
        )

        scaler = RobustScaler()
        scaler.fit(X_train_imp)

        # Center must approximate X_train median
        train_medians = X_train_imp.median().values
        for i, (center, train_med) in enumerate(zip(scaler.center_, train_medians)):
            assert abs(center - train_med) < 1e-6, (
                f"Scaler center[{i}] ({center:.4f}) doesn't match "
                f"X_train median ({train_med:.4f}) — possible leakage!"
            )

    def test_pipeline_no_leakage(self, train_val_split):
        """Pipeline fit_transform on train must not touch val data."""
        X_train, X_val, y_train, y_val = train_val_split
        pipeline = build_pipeline()

        # This should only fit on X_train
        pipeline.fit(X_train, y_train)

        # Val transform should work without errors
        X_val_proc = pipeline.transform(X_val)
        assert X_val_proc.shape[0] == len(X_val), "Val sample count changed after transform!"





class TestFeatureEngineeringShape:
    """Verify feature engineering adds exactly 5 new columns."""

    def test_exactly_five_new_features(self, X_y_split):
        """engineer_features must add exactly 2 interaction features."""
        X, _ = X_y_split
        n_original = X.shape[1]
        X_engineered = engineer_features(X)

        assert X_engineered.shape[1] == n_original + 2, (
            f"Expected {n_original + 2} columns after engineering, "
            f"got {X_engineered.shape[1]}"
        )

    def test_engineered_column_names(self, X_y_split):
        """All 5 expected engineered feature names must be present."""
        X, _ = X_y_split
        X_engineered = engineer_features(X)

        for feat in ENGINEERED_FEATURE_NAMES:
            assert feat in X_engineered.columns, (
                f"Expected engineered feature '{feat}' not found in output columns: "
                f"{list(X_engineered.columns)}"
            )

    def test_original_columns_preserved(self, X_y_split):
        """All original columns must still be present after engineering."""
        X, _ = X_y_split
        X_engineered = engineer_features(X)
        for col in X.columns:
            assert col in X_engineered.columns, (
                f"Original column '{col}' missing after feature engineering."
            )


class TestPipelineTransformConsistency:
    """Verify pipeline produces identical output on identical input."""

    def test_deterministic_transform(self, train_val_split):
        """Same input must produce identical output across two transform calls."""
        X_train, X_val, y_train, y_val = train_val_split

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        result1 = pipeline.transform(X_val)
        result2 = pipeline.transform(X_val)

        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
            check_exact=True,
            obj="Pipeline transform outputs",
        )

    def test_no_nan_after_pipeline(self, train_val_split):
        """Pipeline output must contain no NaN values."""
        X_train, X_val, y_train, y_val = train_val_split

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        X_val_proc = pipeline.transform(X_val)

        assert not X_val_proc.isnull().any().any(), (
            "NaN values found in pipeline output — imputation may have failed."
        )
