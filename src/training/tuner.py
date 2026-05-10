"""
Optuna hyperparameter optimisation for XGBoost on the Water Potability dataset.

Optimises a balanced composite objective on the VALIDATION set only.
Using the test set during optimisation would constitute data leakage.

Objective function design (per Section 8 of the master prompt):
    composite_score = 0.5 * f1 + 0.3 * accuracy + 0.2 * roc_auc

    Why NOT optimise F1 alone:
        F1-only optimisation drives the threshold toward very low values,
        inflating recall at the expense of precision. On a dataset with
        ~39% positive class rate, this produces a model that predicts
        "Potable" for almost everything (recall→1.0, precision→0.39).

    Why TPESampler + MedianPruner:
        TPE (Tree-structured Parzen Estimator) models p(y|x) explicitly
        and explores the hyperparameter space more efficiently than random
        search. MedianPruner stops unpromising trials early using
        intermediate validation scores, reducing total compute time.
"""

from typing import Any, Dict, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.trainer import train_xgboost
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")

# Suppress Optuna's internal info-level noise to reduce log clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_study(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 100,
    random_state: int = 42,
    early_stopping_rounds: int = 30,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna Bayesian optimisation to find the best XGBoost hyperparameters.

    Search space:
        n_estimators:      [100, 500] — upper bound with early stopping active
        max_depth:         [2, 4]     — SHALLOW ONLY (prevents memorisation)
        learning_rate:     [0.01, 0.1] (log scale)
        min_child_weight:  [3, 10]
        subsample:         [0.6, 0.85]
        colsample_bytree:  [0.6, 0.85]
        reg_alpha (L1):    [0.1, 5.0]
        reg_lambda (L2):   [1.0, 10.0]
        gamma:             [0.0, 1.0]

    Evaluation is performed EXCLUSIVELY on the validation set.
    The test set is never used during optimisation.

    Args:
        X_train:               SMOTE-resampled training features.
        y_train:               Resampled training labels.
        X_val:                 Preprocessed validation features.
        y_val:                 Validation labels.
        n_trials:              Number of Optuna trials to run (default 100).
        random_state:          Random seed for reproducibility.
        early_stopping_rounds: Early stopping rounds for each XGBoost trial.

    Returns:
        Tuple of (best_params dict, completed optuna.Study object).
    """
    from sklearn.metrics import matthews_corrcoef

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 2, 3), # STRICT CAP
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 1.5), # Slight oversampling
            "subsample":        trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1.0, 10.0), # HIGH L1
            "reg_lambda":       trial.suggest_float("reg_lambda", 5.0, 20.0), # HIGH L2
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 25), # Prevent small leaves
        }

        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        val_preds = (val_proba >= 0.5).astype(int)

        val_preds = (val_proba >= 0.48).astype(int)
        
        from sklearn.metrics import accuracy_score, f1_score
        f1 = f1_score(y_val, val_preds, zero_division=0)
        acc = accuracy_score(y_val, val_preds)
        
        # Penalize models that drop accuracy below base rate (0.61)
        if acc < 0.61:
            return 0.0
            
        return float(0.5 * f1 + 0.5 * acc)

    sampler = TPESampler(seed=random_state)
    pruner  = MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="water_potability_xgboost",
    )

    logger.info(f"Starting Optuna study — {n_trials} trials, direction=maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Optuna study complete — best composite score: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    return best_params, study
