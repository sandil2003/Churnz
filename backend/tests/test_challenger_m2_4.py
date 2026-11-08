"""
Milestone M2 Challenger Test Suite (challenger_m2_4):
Adversarial Empirical Verification of Explainability Formatting and Database Observability.

Objectives:
1. Stress test format_feature_explanation with extreme and corrupted inputs:
   string floats ('1949.4'), empty strings (''), whitespace ('   '), None, NaN, inf,
   negative infinity, non-numeric strings ('N/A'), dicts, lists.
   Verify that under NO circumstances does format_feature_explanation throw an unhandled exception.
2. Stress test is_db_connected() and /health across state transitions:
   uninitialized DB -> initialized in-memory SQLite -> closed engine.
   Verify database_connected is accurate across state transitions.
3. Verify that load_production_cohort(db_session=None) dynamically instantiates a session
   and queries records when DB is active.
"""

import os
import json
import uuid
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
import src.database as db_mod
from src.database import Base, PredictionLog
import src.api.deps as deps
from src.api.predict import format_feature_explanation
from src.monitoring import load_production_cohort
from src.config import settings
import joblib

client = TestClient(app)


# =========================================================================
# 1. EXPLAINABILITY FORMATTING STRESS TESTS
# =========================================================================

CORRUPTED_VALUES = [
    "1949.4",
    "",
    "   ",
    None,
    float("nan"),
    np.nan,
    float("inf"),
    np.inf,
    float("-inf"),
    -np.inf,
    "N/A",
    {"nested_key": "nested_value"},
    [1, 2, 3],
    12345,
    -99.99,
]

NUMERICAL_FEATURES = ["MonthlyCharges", "TotalCharges", "tenure"]
CATEGORICAL_FEATURES = [
    "Contract_Two year",
    "Contract_One year",
    "Contract",
    "is_month_to_month",
    "tenure_year",
    "has_fiber_optic",
    "InternetService_Fiber optic",
    "InternetService_No",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Credit card (automatic)",
    "has_tech_support",
    "TechSupport_Yes",
    "TechSupport_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_Yes",
    "DeviceProtection_Yes",
    "PaperlessBilling_Yes",
    "avg_monthly_charges",
    "num_services",
    "arbitrary_unmapped_feature",
]


@pytest.mark.parametrize("feature", NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
@pytest.mark.parametrize("corrupted_val", CORRUPTED_VALUES)
def test_format_feature_explanation_corrupted_inputs_in_dict(feature, corrupted_val):
    """
    Stress-tests format_feature_explanation when corrupted / extreme inputs
    are present inside raw_record dictionary.
    Must never crash or throw unhandled exceptions.
    """
    raw_record = {feature: corrupted_val}

    for attribution in [0.85, -0.75, 0.0]:
        explanation = format_feature_explanation(
            feature=feature,
            attribution=attribution,
            raw_record=raw_record
        )
        assert isinstance(explanation, str), f"Expected string explanation, got {type(explanation)}"
        assert len(explanation) > 0, "Explanation must not be empty"


def test_format_feature_explanation_string_float_semantic_output():
    """
    Verifies that string floats like '1949.4' in TotalCharges and '85.5' in MonthlyCharges
    are properly parsed into floating-point format in display strings.
    """
    res_tc_pos = format_feature_explanation("TotalCharges", 0.5, {"TotalCharges": "1949.4"})
    assert "$1949.40" in res_tc_pos
    assert "elevated churn sensitivity" in res_tc_pos

    res_tc_neg = format_feature_explanation("TotalCharges", -0.5, {"TotalCharges": "1949.4"})
    assert "$1949.40" in res_tc_neg
    assert "significant customer commitment" in res_tc_neg

    res_mc_pos = format_feature_explanation("MonthlyCharges", 0.5, {"MonthlyCharges": "85.5"})
    assert "$85.50/mo" in res_mc_pos
    assert "price sensitivity" in res_mc_pos

    res_mc_neg = format_feature_explanation("MonthlyCharges", -0.5, {"MonthlyCharges": "85.5"})
    assert "$85.50/mo" in res_mc_neg
    assert "protect against churn" in res_mc_neg


def test_format_feature_explanation_whitespace_and_nan_fallbacks():
    """
    Verifies that whitespace '   ' or NaN in charges cleanly produces semantic fallbacks
    without format error codes.
    """
    res_tc_space = format_feature_explanation("TotalCharges", 0.5, {"TotalCharges": "   "})
    assert "Accumulated total spend" in res_tc_space
    assert "$" not in res_tc_space

    res_tc_nan = format_feature_explanation("TotalCharges", 0.5, {"TotalCharges": float("nan")})
    assert "Accumulated total spend" in res_tc_nan

    res_mc_space = format_feature_explanation("MonthlyCharges", 0.5, {"MonthlyCharges": "   "})
    assert "High monthly charges" in res_mc_space
    assert "$" not in res_mc_space


def test_format_feature_explanation_falsy_raw_record():
    """
    Verifies format_feature_explanation handles None, empty dict, and empty containers.
    """
    for empty_val in [None, {}, "", []]:
        res = format_feature_explanation("TotalCharges", 0.5, empty_val)
        assert isinstance(res, str)
        assert len(res) > 0


# =========================================================================
# 2. DATABASE OBSERVABILITY & HEALTHCHECK STATE TRANSITIONS
# =========================================================================

def test_database_observability_state_transitions():
    """
    Stress-tests is_db_connected() and GET /health across 3 state transitions:
    1. Uninitialized DB -> database_connected: False
    2. Initialized in-memory SQLite -> database_connected: True
    3. Closed engine -> database_connected: False
    """
    # -------------------------------------------------------------
    # Transition 1: Uninitialized DB
    # -------------------------------------------------------------
    db_mod.engine = None
    db_mod.SessionLocal = None
    deps._last_db_check = 0.0
    deps._cached_db_status = False

    status_uninit = deps.is_db_connected()
    health_uninit = client.get("/health").json()

    assert status_uninit is False, "Uninitialized DB should report False"
    assert health_uninit.get("database_connected") is False, "/health must report False for uninitialized DB"
    assert health_uninit.get("status") in ("healthy", "degraded")

    # -------------------------------------------------------------
    # Transition 2: Initialized in-memory SQLite
    # -------------------------------------------------------------
    deps._last_db_check = 0.0
    init_success = db_mod.init_db_connection("sqlite:///:memory:")
    assert init_success is True, "In-memory SQLite initialization must succeed"
    assert db_mod.SessionLocal is not None, "SessionLocal must be populated"

    status_init = deps.is_db_connected()
    health_init = client.get("/health").json()

    assert status_init is True, "Active SQLite DB should report True"
    assert health_init.get("database_connected") is True, "/health must report True for active DB"

    # -------------------------------------------------------------
    # Transition 3: Closed / Broken Engine
    # -------------------------------------------------------------
    deps._last_db_check = 0.0
    if db_mod.engine is not None:
        db_mod.engine.dispose()

    # Simulate closed connection / broken engine
    broken_engine = create_engine("sqlite:///nonexistent_dir_challenger_test/invalid.db")
    db_mod.SessionLocal = sessionmaker(bind=broken_engine)

    status_closed = deps.is_db_connected()
    health_closed = client.get("/health").json()

    assert status_closed is False, "Closed / broken DB must report False"
    assert health_closed.get("database_connected") is False, "/health must report False for closed engine"

    # Teardown / Reset
    broken_engine.dispose()
    db_mod.engine = None
    db_mod.SessionLocal = None
    deps._last_db_check = 0.0
    deps._cached_db_status = False


# =========================================================================
# 3. PRODUCTION COHORT DYNAMIC SESSION INSTANTIATION
# =========================================================================

def test_load_production_cohort_dynamic_session_instantiation(tmp_path):
    """
    Verifies that load_production_cohort(db_session=None):
    1. Dynamically instantiates a session from db_mod.SessionLocal.
    2. Queries live prediction_logs when count >= min_rows (100).
    3. Transforms records to 36 feature pipeline columns.
    4. Automatically closes the session in finally block without leaking.
    5. Falls back cleanly when count < min_rows.
    """
    db_path = str(tmp_path / "test_cohort.db")
    db_url = f"sqlite:///{db_path}"

    db_mod.init_db_connection(db_url)
    assert db_mod.SessionLocal is not None

    artifact = joblib.load(settings.MODEL_PATH)
    pipeline = artifact["pipeline"]

    # --- Phase A: Sparse DB (<100 records) -> Graceful Fallback ---
    session = db_mod.SessionLocal()
    sample_profile = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 895.0,
    }

    for i in range(10):
        session.add(
            PredictionLog(
                prediction_id=str(uuid.uuid4()),
                customer_id=f"SPARSE-{i:04d}",
                churn_prediction=1,
                churn_probability=0.82,
                risk_level="High",
                features_json=json.dumps(sample_profile),
                model_version="test-v1.0.0",
                latency_ms=12.0
            )
        )
    session.commit()
    session.close()

    # Call with db_session=None -> should dynamically query, see count=10 < 100, and fallback
    cohort_sparse, source_sparse = load_production_cohort(
        db_session=None,
        pipeline=pipeline,
        min_rows=100
    )
    assert "fallback" in source_sparse, f"Expected fallback cohort, got: {source_sparse}"
    assert cohort_sparse is not None
    assert cohort_sparse.shape[1] == 36

    # --- Phase B: Dense DB (>=100 records) -> Live Logs Query ---
    session = db_mod.SessionLocal()
    for i in range(10, 115):
        session.add(
            PredictionLog(
                prediction_id=str(uuid.uuid4()),
                customer_id=f"DENSE-{i:04d}",
                churn_prediction=0 if i % 2 == 0 else 1,
                churn_probability=0.35 if i % 2 == 0 else 0.78,
                risk_level="Low" if i % 2 == 0 else "High",
                features_json=json.dumps(sample_profile),
                model_version="test-v1.0.0",
                latency_ms=9.5
            )
        )
    session.commit()
    session.close()

    # Call with db_session=None -> should dynamically instantiate session and return real logs
    cohort_dense, source_dense = load_production_cohort(
        db_session=None,
        pipeline=pipeline,
        min_rows=100
    )
    assert source_dense == "prediction_logs", f"Expected 'prediction_logs', got: {source_dense}"
    assert len(cohort_dense) == 115, f"Expected 115 records, got: {len(cohort_dense)}"
    assert cohort_dense.shape[1] == 36, f"Expected 36 features, got: {cohort_dense.shape[1]}"

    # Teardown
    db_mod.engine.dispose()
    db_mod.engine = None
    db_mod.SessionLocal = None
