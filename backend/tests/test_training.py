"""
Unit and integration tests for leak-free training, validation early stopping,
and Optuna hyperparameter persistence.
"""
import os
import json
import tempfile
import pandas as pd
import pytest
from src.train import train_model, extract_target, load_best_hyperparams
from src.tune import run_tuning
from src.config import settings


def test_extract_target():
    df = pd.DataFrame({"Churn": ["Yes", "No", "1", "0", " Yes ", "Unknown"]})
    y = extract_target(df)
    assert list(y) == [1, 0, 1, 0, 1, 0]


def test_load_best_hyperparams_missing_file():
    params = load_best_hyperparams("/non/existent/path.json")
    assert params == {}


def test_load_best_hyperparams_valid_file(tmp_path):
    p = tmp_path / "test_params.json"
    dummy = {"num_leaves": 42, "learning_rate": 0.08}
    p.write_text(json.dumps(dummy))
    loaded = load_best_hyperparams(str(p))
    assert loaded["num_leaves"] == 42
    assert loaded["learning_rate"] == 0.08


def test_tuning_and_training_integration(tmp_path):
    # Run a fast 2-trial tuning study to a temporary hyperparams file
    hyperparams_file = str(tmp_path / "best_hyperparams.json")
    best_params = run_tuning(n_trials=2, output_path=hyperparams_file)
    assert os.path.exists(hyperparams_file)
    assert "num_leaves" in best_params

    # Run training
    metrics = train_model()
    assert "val_roc_auc" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.70
    assert os.path.exists(settings.MODEL_PATH)
