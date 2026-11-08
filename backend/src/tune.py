"""
Optuna Hyperparameter Optimization for LightGBM Churn Classifier.
Eliminates data leakage by tuning strictly on training/validation splits
and persists optimal parameters to backend/models/best_hyperparams.json.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from src.config import settings, BASE_DIR
from src.pipeline import TelcoFeaturePipeline

# Suppress verbose Optuna logging for clean CLI output
optuna.logging.set_verbosity(optuna.logging.INFO)


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Extracts and binarizes target variable 'Churn'."""
    churn_col = "Churn" if "Churn" in df.columns else ("churn" if "churn" in df.columns else None)
    if churn_col is None:
        raise ValueError("Target column 'Churn' missing from dataset.")
    churn_series = df[churn_col].astype(str).str.strip()
    return churn_series.map({
        "Yes": 1, "No": 0, "1": 1, "0": 0,
        "True": 1, "False": 0, "true": 1, "false": 0
    }).fillna(0).astype(int)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function with 3-Fold Stratified Cross Validation."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 80, 250),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbose": -1
    }

    # 3-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr)
        model = lgb.LGBMClassifier(**params, scale_pos_weight=float(scale_pos))
        model.fit(X_tr, y_tr)

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return float(pd.Series(scores).mean())


def run_tuning(
    n_trials: int = 15,
    data_path: str = settings.DATA_RAW_PATH,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs Optuna hyperparameter study without data leakage.
    Splits out holdout test data first, fits pipeline on training split,
    runs cross-validation, and saves optimal hyperparameters to JSON.
    """
    if output_path is None:
        output_path = getattr(
            settings,
            "HYPERPARAMS_PATH",
            str(BASE_DIR / "models" / "best_hyperparams.json")
        )

    print(f"--> Starting Optuna hyperparameter study ({n_trials} trials)...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    churn_col = "Churn" if "Churn" in df.columns else ("churn" if "churn" in df.columns else None)
    if churn_col is None:
        raise ValueError("Target column 'Churn' missing from dataset.")

    # 1. Leak-Free Split: Carve out 20% holdout test set before feature engineering
    train_val_df, _ = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[churn_col]
    )

    # 2. Fit pipeline ONLY on training/validation pool
    pipeline = TelcoFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_val_df)

    # 3. Optimize Hyperparameters
    study = optuna.create_study(direction="maximize", study_name="lgbm_churn_tuning")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    best_value = float(study.best_value)

    print("\n========== OPTUNA STUDY RESULTS ==========")
    print(f"Best Trial ROC-AUC: {best_value:.4f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("==========================================\n")

    # 4. Save best hyperparameters to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"--> Saved best hyperparameters to: {output_path}")

    return best_params


if __name__ == "__main__":
    run_tuning(n_trials=15)
