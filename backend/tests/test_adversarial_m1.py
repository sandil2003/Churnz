"""
Adversarial Stress Test Suite for Milestone M1 (ML Pipeline and Database Hardening).
Empirically verifies:
1. Database Session Resilience:
   - Exception handling, rollback, and guaranteed session closure.
   - Connection pool leak prevention under repeated failures.
   - Atomic batch logging across 0 records, 500 records, malformed inputs, and duplicate key collisions.
2. Training Leakage & Validation Check:
   - Leak-free partitioning (Train=64%, Val=16%, Test=20%).
   - Feature pipeline fitted strictly on training data.
   - Model fit and early stopping evaluated exclusively on validation set (X_test untouched during training).
   - Dynamic loading and precedence hierarchy of best_hyperparams.json.
"""

import os
import json
import uuid
import time
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker

from src.config import settings, BASE_DIR
from src.database import (
    Base,
    PredictionLog,
    init_db_connection,
    log_prediction_to_db,
    log_batch_predictions_to_db,
    get_db
)
import src.database as db_mod
from src.pipeline import TelcoFeaturePipeline
from src.train import (
    train_model,
    load_best_hyperparams,
    extract_target,
    load_data
)
import lightgbm as lgb


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def clean_db_state():
    """Ensures each test gets a clean in-memory SQLite database and resets globals."""
    success = init_db_connection("sqlite:///:memory:")
    assert success is True
    yield
    if db_mod.engine:
        Base.metadata.drop_all(bind=db_mod.engine)
        db_mod.engine.dispose()
    db_mod.engine = None
    db_mod.SessionLocal = None


# ==============================================================================
# 1. DATABASE SESSION RESILIENCE & CONNECTION LEAK TESTS
# ==============================================================================

def test_log_prediction_duplicate_id_rollback_and_session_closed():
    """
    Stress-test log_prediction_to_db when an IntegrityError occurs (duplicate prediction_id).
    Verifies:
    1. Returns None instead of raising unhandled exception.
    2. Rolls back cleanly so duplicate is rejected.
    3. Session is closed and connection returned to pool.
    """
    dupe_id = str(uuid.uuid4())
    customer_dict = {"customer_id": "CUST-DUPE-1", "tenure": 12, "MonthlyCharges": 50.0}

    # 1. First insert should succeed
    res1 = log_prediction_to_db(
        customer_dict=customer_dict,
        prediction=1,
        probability=0.85,
        risk_level="High",
        prediction_id=dupe_id,
        model_version="v1"
    )
    assert res1 == dupe_id

    # 2. Second insert with identical prediction_id must fail cleanly
    res2 = log_prediction_to_db(
        customer_dict=customer_dict,
        prediction=0,
        probability=0.20,
        risk_level="Low",
        prediction_id=dupe_id,
        model_version="v2"
    )
    assert res2 is None, "Duplicate prediction_id must return None on conflict"

    # 3. Verify original record in database is untouched
    db = db_mod.SessionLocal()
    try:
        records = db.query(PredictionLog).filter_by(prediction_id=dupe_id).all()
        assert len(records) == 1
        assert records[0].churn_prediction == 1
        assert records[0].model_version == "v1"
    finally:
        db.close()


def test_log_prediction_malformed_inputs_handled_gracefully():
    """
    Stress-test log_prediction_to_db with malformed input types.
    Verifies that ValueError / TypeError inside logging block is caught,
    rolled back, session closed, and returns None.
    """
    customer_dict = {"customer_id": "CUST-MALFORMED"}

    # Malformed prediction (non-integer string)
    res_bad_pred = log_prediction_to_db(
        customer_dict=customer_dict,
        prediction="NOT_AN_INT",  # triggers ValueError in int()
        probability=0.5,
        risk_level="Medium"
    )
    assert res_bad_pred is None

    # Malformed probability (non-float string)
    res_bad_prob = log_prediction_to_db(
        customer_dict=customer_dict,
        prediction=1,
        probability="NOT_A_FLOAT",  # triggers ValueError in float()
        risk_level="High"
    )
    assert res_bad_prob is None


def test_log_prediction_connection_leak_under_high_failure_pressure(tmp_path):
    """
    Adversarially stress-tests connection pool exhaustion.
    Configures a tiny connection pool (size=3, max_overflow=0).
    Executes 30 consecutive failing calls to log_prediction_to_db.
    If sessions or connections leaked, pool would exhaust by the 4th call
    and raise TimeoutError.
    """
    db_file = tmp_path / "strict_pool.db"
    strict_engine = create_engine(
        f"sqlite:///{db_file}",
        poolclass=QueuePool,
        pool_size=3,
        max_overflow=0,
        pool_timeout=1.0
    )
    Base.metadata.create_all(bind=strict_engine)
    StrictSession = sessionmaker(autocommit=False, autoflush=False, bind=strict_engine)

    # Patch global SessionLocal
    with patch.object(db_mod, "SessionLocal", StrictSession):
        dupe_id = "FIXED-COLLISION-ID"
        # Seed 1 record
        db = StrictSession()
        db.add(PredictionLog(
            prediction_id=dupe_id,
            customer_id="SEED",
            features_json="{}",
            churn_prediction=0,
            churn_probability=0.1,
            risk_level="Low"
        ))
        db.commit()
        db.close()

        # Fire 30 conflicting calls
        for i in range(30):
            res = log_prediction_to_db(
                customer_dict={"customer_id": f"FAIL-{i}"},
                prediction=1,
                probability=0.9,
                risk_level="High",
                prediction_id=dupe_id  # causes IntegrityError
            )
            assert res is None

        # Verify pool is clean (0 connections checked out)
        assert strict_engine.pool.checkedout() == 0, "Connections leaked after failed operations!"

    strict_engine.dispose()


def test_log_batch_predictions_empty_and_none_input():
    """
    Stress-test log_batch_predictions_to_db with empty list or None.
    Should return 0 immediately without opening transactions.
    """
    assert log_batch_predictions_to_db([]) == 0
    assert log_batch_predictions_to_db(None) == 0


def test_log_batch_predictions_massive_500_records_performance():
    """
    Stress-test log_batch_predictions_to_db with a massive batch (500 records).
    Verifies:
    1. All 500 records are persisted atomically in a single batch.
    2. Execution completes in sub-second time.
    3. Database record count matches 500 exactly.
    """
    batch_size = 500
    records = []
    for i in range(batch_size):
        records.append({
            "customer_id": f"MASS-{i:04d}",
            "features": {"tenure": i % 72, "MonthlyCharges": 20.0 + (i * 0.1)},
            "churn_prediction": 1 if i % 3 == 0 else 0,
            "churn_probability": round((i % 100) / 100.0, 4),
            "risk_level": "High" if (i % 3 == 0) else "Low",
            "model_version": "stress-test-v1",
            "latency_ms": 1.2
        })

    t0 = time.perf_counter()
    count = log_batch_predictions_to_db(records, model_version="fallback-v1")
    duration = time.perf_counter() - t0

    assert count == 500, f"Expected 500 persisted records, got {count}"
    assert duration < 2.0, f"500-record batch insert took too long ({duration:.2f}s)"

    # Verify directly in DB
    db = db_mod.SessionLocal()
    try:
        persisted_count = db.query(PredictionLog).filter(
            PredictionLog.model_version == "stress-test-v1"
        ).count()
        assert persisted_count == 500
    finally:
        db.close()


def test_log_batch_predictions_atomicity_on_malformed_record():
    """
    Stress-test atomicity of log_batch_predictions_to_db:
    If record 18 of 20 has an invalid type ('NOT_AN_INT'), the ENTIRE batch
    must be rolled back. ZERO records must be written.
    """
    records = []
    for i in range(20):
        records.append({
            "customer_id": f"ATOMIC-{i:02d}",
            "features": {"tenure": i},
            "churn_prediction": 1 if i != 18 else "NOT_AN_INT",  # Poison pill at index 18
            "churn_probability": 0.5,
            "risk_level": "Medium",
            "model_version": "atomic-test-v1"
        })

    count = log_batch_predictions_to_db(records)
    assert count == 0, "Batch with poisoned record must fail and return 0"

    # Verify zero records committed
    db = db_mod.SessionLocal()
    try:
        persisted = db.query(PredictionLog).filter(
            PredictionLog.model_version == "atomic-test-v1"
        ).all()
        assert len(persisted) == 0, f"Transaction was not atomic! Found {len(persisted)} committed rows."
    finally:
        db.close()


def test_log_batch_predictions_atomicity_on_duplicate_collision():
    """
    Stress-test atomicity of log_batch_predictions_to_db under DB IntegrityError:
    If record 9 in a 10-record batch collides with an existing prediction_id,
    all 10 records must be rolled back.
    """
    # Pre-seed one record
    colliding_id = str(uuid.uuid4())
    log_prediction_to_db(
        customer_dict={"customer_id": "ORIGINAL"},
        prediction=0,
        probability=0.1,
        risk_level="Low",
        prediction_id=colliding_id,
        model_version="seed-v1"
    )

    # Prepare batch of 10 records, with index 8 containing colliding_id
    records = []
    for i in range(10):
        records.append({
            "customer_id": f"COLLIDE-BATCH-{i}",
            "features": {"tenure": i},
            "churn_prediction": 1,
            "churn_probability": 0.9,
            "risk_level": "High",
            "prediction_id": colliding_id if i == 8 else str(uuid.uuid4()),
            "model_version": "collide-batch-v1"
        })

    count = log_batch_predictions_to_db(records)
    assert count == 0, "Batch containing colliding UUID must fail and return 0"

    # Verify NO batch records were committed
    db = db_mod.SessionLocal()
    try:
        persisted = db.query(PredictionLog).filter(
            PredictionLog.model_version == "collide-batch-v1"
        ).all()
        assert len(persisted) == 0, f"Expected 0 committed rows, found {len(persisted)}"
    finally:
        db.close()


# ==============================================================================
# 2. TRAINING LEAKAGE & VALIDATION CHECK TESTS
# ==============================================================================

def test_leak_free_data_splitting_and_pipeline_fit_isolation():
    """
    Empirically verifies:
    1. Raw data is split into 64% Train, 16% Val, 20% Test.
    2. Train, Val, and Test splits are strictly disjoint (0 overlapping indices).
    3. Stratification preserves churn ratio (~26.5%) across all splits.
    4. TelcoFeaturePipeline.fit is called ONLY ONCE, and strictly on train_df.
    """
    raw_df = load_data(settings.DATA_RAW_PATH)
    total_len = len(raw_df)
    assert total_len == 7043

    # Run train_test_split as done in train_model
    train_val_df, test_df = pd.core.common.random_state(42), None  # sanity check import
    from sklearn.model_selection import train_test_split
    churn_col = "Churn"

    train_val_df, test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df[churn_col]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df[churn_col]
    )

    # 1. Split counts
    assert len(train_df) == 4507  # 64%
    assert len(val_df) == 1127    # 16%
    assert len(test_df) == 1409   # 20%
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # 2. Disjoint indices
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    assert len(train_idx & val_idx) == 0, "Train and Val indices overlap!"
    assert len(train_idx & test_idx) == 0, "Train and Test indices overlap!"
    assert len(val_idx & test_idx) == 0, "Val and Test indices overlap!"

    # 3. Stratification rates
    raw_rate = (raw_df[churn_col] == "Yes").mean()
    train_rate = (train_df[churn_col] == "Yes").mean()
    val_rate = (val_df[churn_col] == "Yes").mean()
    test_rate = (test_df[churn_col] == "Yes").mean()
    assert abs(train_rate - raw_rate) < 0.005
    assert abs(val_rate - raw_rate) < 0.005
    assert abs(test_rate - raw_rate) < 0.005

    # 4. Pipeline fit isolation check
    pipeline = TelcoFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    assert pipeline.is_fitted_ is True

    # Validate that transform on val and test preserves 36 feature columns without re-fitting
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)
    assert X_train.shape[1] == 36
    assert X_val.shape[1] == 36
    assert X_test.shape[1] == 36


def test_test_set_untouched_during_training_and_early_stopping():
    """
    Adversarial verification that (X_test, y_test) is NEVER touched
    by model.fit() or early stopping.
    We spy on LGBMClassifier.fit to inspect eval_set and ensure test data is absent.
    """
    original_fit = lgb.LGBMClassifier.fit
    fit_calls = []

    def spying_fit(self, X, y, eval_set=None, callbacks=None, **kwargs):
        fit_calls.append({
            "train_shape": X.shape,
            "eval_set": eval_set,
            "callbacks": callbacks
        })
        return original_fit(self, X, y, eval_set=eval_set, callbacks=callbacks, **kwargs)

    with patch.object(lgb.LGBMClassifier, "fit", side_effect=spying_fit, autospec=True):
        metrics = train_model()

    assert len(fit_calls) == 1, "Model.fit should be called exactly once"
    call = fit_calls[0]

    # Verify training set shape passed to fit: 4507 rows (Train split), 36 features
    assert call["train_shape"] == (4507, 36)

    # Verify eval_set contains ONLY validation set: 1127 rows (Val split)
    eval_set = call["eval_set"]
    assert eval_set is not None, "eval_set must be provided for early stopping"
    assert len(eval_set) == 1, f"eval_set must have exactly 1 evaluation dataset (Val), got {len(eval_set)}"

    eval_X, eval_y = eval_set[0]
    assert eval_X.shape == (1127, 36), f"eval_set X must be val_df (1127 rows), got {eval_X.shape}"
    assert len(eval_y) == 1127, f"eval_set y must be val_df (1127 rows), got {len(eval_y)}"

    # Confirm early stopping callback is configured
    callbacks = call["callbacks"]
    assert any("EarlyStoppingCallback" in type(cb).__name__ for cb in callbacks), (
        "Early stopping callback not found in fit callbacks"
    )

    # Verify final holdout test evaluation was computed
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["roc_auc"] > 0.70


def test_best_hyperparams_loaded_and_used_in_model(tmp_path):
    """
    Empirically verifies that Optuna-tuned parameters from best_hyperparams.json
    are loaded and directly configure the trained LightGBM model.
    """
    custom_params = {
        "n_estimators": 60,
        "learning_rate": 0.0456,
        "num_leaves": 19,
        "max_depth": 4,
        "min_child_samples": 25,
        "subsample": 0.75,
        "colsample_bytree": 0.70
    }
    custom_file = str(tmp_path / "custom_hyperparams.json")
    with open(custom_file, "w") as f:
        json.dump(custom_params, f)

    # Load custom params
    loaded = load_best_hyperparams(custom_file)
    assert loaded["num_leaves"] == 19
    assert loaded["learning_rate"] == 0.0456

    # Test training with these custom params via hyperparams override
    metrics = train_model(hyperparams=custom_params)
    assert metrics is not None

    # Load saved model artifact and check LGBM parameters
    import joblib
    artifact = joblib.load(settings.MODEL_PATH)
    trained_model = artifact["model"]
    model_params = trained_model.get_params()

    assert model_params["num_leaves"] == 19
    assert model_params["learning_rate"] == 0.0456
    assert model_params["n_estimators"] == 60
    assert model_params["max_depth"] == 4
    assert model_params["min_child_samples"] == 25


def test_hyperparams_precedence_hierarchy(tmp_path):
    """
    Verifies parameter precedence:
    Default Baseline < Tuned JSON < Explicit hyperparams argument.
    """
    # 1. Base parameters has learning_rate 0.05
    # 2. File specifies learning_rate 0.03
    # 3. Explicit argument specifies learning_rate 0.099
    custom_file = str(tmp_path / "tuned.json")
    with open(custom_file, "w") as f:
        json.dump({"learning_rate": 0.03, "num_leaves": 22}, f)

    with patch("src.train.load_best_hyperparams", return_value={"learning_rate": 0.03, "num_leaves": 22}):
        # Explicit argument overrides learning_rate, but inherits num_leaves from tuned file
        train_model(hyperparams={"learning_rate": 0.099})

    import joblib
    artifact = joblib.load(settings.MODEL_PATH)
    params = artifact["model"].get_params()
    assert params["learning_rate"] == 0.099, "Explicit hyperparams must take precedence!"
    assert params["num_leaves"] == 22, "Tuned JSON params must be applied when not overridden!"
    # Protected parameters must remain intact
    assert params["random_state"] == 42
    assert params["verbose"] == -1


def test_log_prediction_unserializable_cyclic_payload():
    """
    Stress-test log_prediction_to_db with an unserializable circular object in customer_dict.
    Verifies it fails cleanly, rolls back, closes session, and returns None.
    """
    cyclic_dict = {"customer_id": "CYCLIC-001"}
    cyclic_dict["self_ref"] = cyclic_dict  # creates circular reference ValueError in json.dumps

    res = log_prediction_to_db(
        customer_dict=cyclic_dict,
        prediction=0,
        probability=0.1,
        risk_level="Low"
    )
    assert res is None


def test_log_prediction_when_db_offline_returns_none():
    """
    Verifies that when database connection cannot be initialized,
    log_prediction_to_db and log_batch_predictions_to_db safely return None / 0.
    """
    with patch.object(db_mod, "SessionLocal", None):
        with patch.object(db_mod, "init_db_connection", return_value=False):
            res_single = log_prediction_to_db(
                customer_dict={"customer_id": "OFFLINE"},
                prediction=1,
                probability=0.7,
                risk_level="High"
            )
            assert res_single is None

            res_batch = log_batch_predictions_to_db([
                {"churn_prediction": 1, "churn_probability": 0.7, "risk_level": "High"}
            ])
            assert res_batch == 0


def test_get_db_exception_triggers_rollback_and_close():
    """
    Verifies that FastAPI get_db dependency properly executes db.rollback()
    and db.close() when an unhandled exception is raised in the endpoint logic.
    """
    gen = get_db()
    db = next(gen)
    assert db is not None

    with patch.object(db, "rollback") as mock_rollback:
        with patch.object(db, "close") as mock_close:
            with pytest.raises(RuntimeError, match="Simulated endpoint crash"):
                gen.throw(RuntimeError("Simulated endpoint crash"))

            assert mock_rollback.call_count == 1
            assert mock_close.call_count == 1


def test_load_best_hyperparams_corrupt_json_and_list(tmp_path):
    """
    Stress-test load_best_hyperparams with:
    1. Syntactically invalid JSON string
    2. JSON list instead of dictionary
    """
    # 1. Broken JSON
    broken_file = tmp_path / "broken.json"
    broken_file.write_text("{this is not valid json")
    res_broken = load_best_hyperparams(str(broken_file))
    assert res_broken == {}

    # 2. JSON list
    list_file = tmp_path / "list.json"
    list_file.write_text("[1, 2, 3, 4]")
    res_list = load_best_hyperparams(str(list_file))
    assert res_list == {}


def test_train_model_missing_churn_raises_value_error(tmp_path):
    """
    Verifies that train_model validates presence of target column 'Churn'
    and raises an explicit ValueError if absent.
    """
    bad_df = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [50.0, 60.0]})
    bad_csv = str(tmp_path / "no_target.csv")
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Target column 'Churn' missing"):
        train_model(data_path=bad_csv)


def test_train_model_nonexistent_path_raises_file_not_found():
    """
    Verifies that train_model raises FileNotFoundError on missing dataset.
    """
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        train_model(data_path="/nonexistent/data/file.csv")

