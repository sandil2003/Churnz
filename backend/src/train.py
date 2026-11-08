"""
Production Training Pipeline for LightGBM Customer Churn Classifier.
Implements leak-free 3-way dataset splitting, true validation early stopping,
Optuna hyperparameter integration, and MLflow experiment tracking.
"""

import os
import json
import socket
from urllib.parse import urlparse
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import mlflow

from src.config import settings, BASE_DIR
from src.pipeline import TelcoFeaturePipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw customer data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Extracts and maps 'Churn' target strings to binary integers (0/1)."""
    churn_col = "Churn" if "Churn" in df.columns else ("churn" if "churn" in df.columns else None)
    if churn_col is None:
        raise ValueError("Target column 'Churn' missing from dataset.")
    churn_series = df[churn_col].astype(str).str.strip()
    return churn_series.map({
        "Yes": 1, "No": 0, "1": 1, "0": 0,
        "True": 1, "False": 0, "true": 1, "false": 0
    }).fillna(0).astype(int)


def load_best_hyperparams(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads Optuna-discovered hyperparameters from JSON file if present.
    Supports both flat dictionaries and wrapped {"best_params": ...} schemas.
    """
    if filepath is None:
        filepath = getattr(
            settings,
            "HYPERPARAMS_PATH",
            str(BASE_DIR / "models" / "best_hyperparams.json")
        )
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                params = data.get("best_params", data)
                print(f"--> Loaded tuned hyperparameters from: {filepath}")
                return params
        except Exception as e:
            print(f"[Warning] Failed to load hyperparameters from {filepath}: {e}")
    return {}


def is_remote_mlflow_reachable(uri: str, timeout: float = 0.5) -> bool:
    """Fast socket check to prevent long urllib3 connection timeouts if server is offline."""
    try:
        parsed = urlparse(uri)
        if parsed.scheme in ("http", "https"):
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=timeout):
                return True
        return True
    except Exception:
        return False


def train_model(
    data_path: str = settings.DATA_RAW_PATH,
    hyperparams: Optional[Dict[str, Any]] = None,
    experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME
) -> Dict[str, Any]:
    """
    Trains champion LightGBM model with:
    1. Leak-free 3-way split: 64% Train / 16% Val / 20% Holdout Test.
    2. Feature engineering pipeline fitted exclusively on train_df.
    3. Early stopping evaluated exclusively on val_df.
    4. Generalization performance evaluated on isolated test_df.
    5. Tuned hyperparameters loaded from backend/models/best_hyperparams.json.
    """
    print(f"--> Loading raw dataset from: {data_path}...")
    raw_df = load_data(data_path)
    churn_col = "Churn" if "Churn" in raw_df.columns else ("churn" if "churn" in raw_df.columns else None)
    if churn_col is None:
        raise ValueError("Target column 'Churn' missing from training dataset.")

    # 1. Leak-Free Partitioning (Raw DataFrame splitting FIRST)
    # Split out 20% final holdout test set
    train_val_df, test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df[churn_col]
    )
    # Carve out 20% validation split from train_val_df (16% of total data)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df[churn_col]
    )

    print(f"--> Dataset partitioned: Train={len(train_df)} | Val={len(val_df)} | Holdout Test={len(test_df)}")

    # 2. Fit feature pipeline ONLY on train_df
    pipeline = TelcoFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)

    # Transform validation and test splits without refitting
    X_val = pipeline.transform(val_df)
    y_val = extract_target(val_df)

    X_test = pipeline.transform(test_df)
    y_test = extract_target(test_df)

    # 3. Calculate class balance weight on training labels
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # Base baseline hyperparameters
    model_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
        "verbose": -1
    }

    # Overlay Optuna tuned hyperparameters from disk if present
    tuned_disk_params = load_best_hyperparams()
    if tuned_disk_params:
        model_params.update(tuned_disk_params)

    # Overlay user-provided explicit overrides if passed
    if hyperparams:
        model_params.update(hyperparams)

    # Preserve dynamic class balance and fixed control settings
    model_params["scale_pos_weight"] = float(scale_pos_weight)
    model_params["random_state"] = 42
    model_params["verbose"] = -1

    # 4. Configure MLflow tracking
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_file = str(BASE_DIR / "mlflow.db").replace("\\", "/")
    local_db_uri = f"sqlite:///{db_file}"

    remote_uri = settings.MLFLOW_TRACKING_URI
    if is_remote_mlflow_reachable(remote_uri):
        mlflow.set_tracking_uri(remote_uri)
    else:
        print(f"[Warning] Remote MLflow server ({remote_uri}) unreachable. Using local SQLite: {local_db_uri}")
        mlflow.set_tracking_uri(local_db_uri)

    try:
        mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run(run_name="lightgbm_champion")
    except Exception as e:
        print(f"[Warning] MLflow server failed ({mlflow.get_tracking_uri()}): {e}. Falling back to local SQLite.")
        mlflow.set_tracking_uri(local_db_uri)
        mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run(run_name="lightgbm_champion")

    with run_ctx as run:
        active_run_id = run.info.run_id if run and hasattr(run, "info") else "local_champion"
        print("--> Training LightGBM Classifier with validation early stopping...")
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # 5. Evaluate on Validation Set
        val_prob = model.predict_proba(X_val)[:, 1]
        val_roc_auc = roc_auc_score(y_val, val_prob)

        # 6. Evaluate on True Holdout Test Set (Unbiased)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, test_prob)
        pr_auc = average_precision_score(y_test, test_prob)
        accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)
        cm = confusion_matrix(y_test, test_pred).tolist()

        metrics = {
            "val_roc_auc": round(float(val_roc_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "best_iteration": int(model.best_iteration_ or model_params.get("n_estimators", 150))
        }

        # Log parameters & metrics to MLflow
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)

        # 7. Extract Feature Importances
        importance_df = pd.DataFrame({
            "feature": pipeline.feature_columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        top_features = importance_df.head(10).to_dict(orient="records")

        # 8. Export Champion Model Artifact
        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
        model_version = f"champion-{active_run_id[:8]}"
        model_artifact = {
            "model": model,
            "pipeline": pipeline,
            "metrics": metrics,
            "top_features": top_features,
            "feature_columns": pipeline.feature_columns,
            "run_id": active_run_id,
            "model_version": model_version
        }
        joblib.dump(model_artifact, settings.MODEL_PATH)
        print(f"--> Saved champion model package to: {settings.MODEL_PATH}")

        # 9. Publish & Register Champion Model to MLflow Models Tab
        try:
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=settings.MLFLOW_MODEL_NAME
            )
            mlflow.log_artifact(settings.MODEL_PATH, artifact_path="model_package")
            client = mlflow.tracking.MlflowClient()
            reg_versions = client.get_latest_versions(settings.MLFLOW_MODEL_NAME)
            if reg_versions:
                latest_v = reg_versions[0].version
                client.set_registered_model_alias(settings.MLFLOW_MODEL_NAME, settings.MLFLOW_MODEL_ALIAS, latest_v)
                print(f"--> Published model '{settings.MLFLOW_MODEL_NAME}' v{latest_v} to MLflow Models tab with alias @{settings.MLFLOW_MODEL_ALIAS}.")
        except Exception as e:
            print(f"[Warning] Could not register model in MLflow Models tab: {e}")

        # 10. Export Reference Data for Evidently AI Monitoring
        reference_data = X_train.copy()
        reference_data["target"] = y_train
        reference_data["prediction"] = model.predict(X_train)
        reference_path = os.path.join(settings.DATA_PROCESSED_PATH, "reference_data.csv")
        os.makedirs(settings.DATA_PROCESSED_PATH, exist_ok=True)
        reference_data.to_csv(reference_path, index=False)
        print(f"--> Saved Evidently AI reference dataset ({len(reference_data)} rows) to: {reference_path}")

        print(
            f"--> Training Complete! Best Iter: {metrics['best_iteration']} | "
            f"Val AUC: {val_roc_auc:.4f} | Test ROC-AUC: {roc_auc:.4f} | "
            f"Test PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}"
        )
        return metrics


if __name__ == "__main__":
    train_model()
