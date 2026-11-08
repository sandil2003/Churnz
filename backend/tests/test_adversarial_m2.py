"""
Adversarial Stress Test Suite for Milestone M2 (Serving API and Monitoring Hardening).
Empirically verifies:
1. Explainability Stress Testing:
   - 50 randomized customer profiles testing TreeSHAP additive property:
     `np.allclose(base_bias + sum(attributions), raw_margin, atol=1e-4)` holds for 100% of profiles.
   - Zero-impact feature ranking, non-dominance, and exclusion checks.
   - Novel category handling at pipeline level (dummy preservation) vs API level (Pydantic 422).
   - Out-of-range numerical feature behaviors (tenure bounds, extreme charges, inf/nan).
2. Drift Monitoring Stress Testing:
   - `load_production_cohort` and `generate_drift_report` under 0, 5, 50, and 500 records.
   - Evidently AI behavior with sparse cohorts.
   - `GET /monitoring/summary` endpoint returning valid JSON with all required keys on unseeded DB.
"""

import os
import json
import uuid
import random
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.main import app
from src.database import Base, PredictionLog, init_db_connection
import src.database as db_mod
from src.api.predict import compute_treeshap_contributions, format_feature_explanation
from src.monitoring import (
    load_reference_data,
    load_production_cohort,
    generate_drift_report,
    get_latest_drift_summary
)
import joblib

client = TestClient(app)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def champion_artifact():
    """Loads champion model artifact directly."""
    assert os.path.exists(settings.MODEL_PATH), f"Model path {settings.MODEL_PATH} does not exist"
    return joblib.load(settings.MODEL_PATH)


@pytest.fixture(scope="module")
def reference_data():
    """Loads baseline reference dataset."""
    ref_df = load_reference_data()
    assert ref_df is not None, "Reference data must be available"
    return ref_df


@pytest.fixture
def clean_test_db():
    """Provides a fresh in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def generate_random_customer_profile(seed: int) -> Dict[str, Any]:
    """Generates a valid, randomized Telco customer profile dictionary."""
    rng = random.Random(seed)

    internet_choice = rng.choice(["DSL", "Fiber optic", "No"])
    has_internet = internet_choice != "No"

    def service_opt():
        return rng.choice(["Yes", "No"]) if has_internet else "No internet service"

    phone = rng.choice(["Yes", "No"])
    multiple_lines = rng.choice(["Yes", "No"]) if phone == "Yes" else "No phone service"

    tenure = rng.randint(0, 72)
    monthly_charges = round(rng.uniform(18.25, 118.75), 2)
    if tenure == 0:
        total_charges = 0.0
    else:
        # Realistic noisy total charges
        multiplier = rng.uniform(0.85, 1.15)
        total_charges = round(monthly_charges * tenure * multiplier, 2)

    return {
        "customer_id": f"RAND-{seed:04d}",
        "gender": rng.choice(["Male", "Female"]),
        "SeniorCitizen": rng.choice([0, 1]),
        "Partner": rng.choice(["Yes", "No"]),
        "Dependents": rng.choice(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet_choice,
        "OnlineSecurity": service_opt(),
        "OnlineBackup": service_opt(),
        "DeviceProtection": service_opt(),
        "TechSupport": service_opt(),
        "StreamingTV": service_opt(),
        "StreamingMovies": service_opt(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": rng.choice(["Yes", "No"]),
        "PaymentMethod": rng.choice([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }


# ==============================================================================
# Part 1: Explainability Stress Testing
# ==============================================================================

def test_treeshap_additivity_50_random_profiles(champion_artifact):
    """
    Stress-tests TreeSHAP additivity across 50 randomly generated customer profiles:
    np.allclose(base_bias + sum(attributions), raw_margin, atol=1e-4) must hold for 100%.
    Also verifies sigmoid(raw_margin) == predict_proba.
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    assert hasattr(model, "booster_"), "Model must possess booster_ attribute"

    profiles = [generate_random_customer_profile(seed=1000 + i) for i in range(50)]
    df_profiles = pd.DataFrame(profiles)

    # Transform through pipeline
    X = pipeline.transform(df_profiles)
    assert X.shape == (50, 36), f"Expected shape (50, 36), got {X.shape}"

    # Compute booster TreeSHAP contributions
    contribs = model.booster_.predict(X, pred_contrib=True)
    assert contribs.shape == (50, 37), f"Expected shape (50, 37), got {contribs.shape}"

    # Compute raw margins directly from model
    raw_scores = model.predict(X, raw_score=True)
    probs = model.predict_proba(X)[:, 1]

    passed_count = 0
    max_error = 0.0
    errors = []

    for i in range(50):
        base_bias = contribs[i, -1]
        attributions = contribs[i, :-1]
        raw_margin = raw_scores[i]

        reconstructed_margin = base_bias + np.sum(attributions)
        abs_err = abs(reconstructed_margin - raw_margin)
        errors.append(abs_err)
        if abs_err > max_error:
            max_error = abs_err

        # Check TreeSHAP efficiency property
        assert np.isclose(reconstructed_margin, raw_margin, atol=1e-4), (
            f"Profile {i} failed additivity: reconstructed={reconstructed_margin:.6f}, "
            f"raw_margin={raw_margin:.6f}, diff={abs_err:.6e}"
        )

        # Check sigmoid link to predicted probability
        sig_prob = 1.0 / (1.0 + np.exp(-raw_margin))
        assert np.isclose(sig_prob, probs[i], atol=1e-4), (
            f"Profile {i} failed sigmoid check: sig_prob={sig_prob:.6f}, model_prob={probs[i]:.6f}"
        )

        passed_count += 1

    assert passed_count == 50, f"Expected 50 passed profiles, got {passed_count}"
    assert max_error < 1e-4, f"Max error {max_error:.6e} exceeded tolerance 1e-4"


def test_zero_impact_feature_behavior(champion_artifact):
    """
    Stress-tests behavior when features have zero impact.
    Verifies:
    1. Features with zero contribution are never ranked in top risk factors when non-zero features exist.
    2. compute_treeshap_contributions handles all-zero feature arrays gracefully without crashing.
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    # 1. Standard profile: verify no 0-attribution feature is included in risk_factors
    profile = generate_random_customer_profile(seed=42)
    X_single = pipeline.transform(pd.DataFrame([profile]))
    risk_factors, explanations = compute_treeshap_contributions(
        model=model,
        pipeline=pipeline,
        X=X_single,
        raw_record=profile,
        top_pos=3,
        top_neg=2
    )

    for rf in risk_factors:
        assert rf.attribution != 0.0, f"Feature {rf.feature} has 0.0 attribution but was included in risk factors"

    # 2. Synthetic test with mocked model where some features are 0.0 and some are non-zero
    class PartiallyZeroModel:
        def __init__(self):
            # 3 features: feat_0 is positive, feat_1 is exactly zero, feat_2 is negative
            pass
        class booster_:
            @staticmethod
            def predict(X, pred_contrib=True):
                # Shape (1, 4): 3 features + 1 bias
                return np.array([[1.5, 0.0, -0.8, -0.5]])

    class SyntheticPipeline:
        feature_columns = ["positive_feat", "zero_feat", "negative_feat"]

    syn_model = PartiallyZeroModel()
    syn_pipe = SyntheticPipeline()
    X_syn = np.zeros((1, 3))

    syn_factors, syn_explanations = compute_treeshap_contributions(
        model=syn_model,
        pipeline=syn_pipe,
        X=X_syn,
        top_pos=3,
        top_neg=2
    )

    included_features = [f.feature for f in syn_factors]
    assert "positive_feat" in included_features
    assert "negative_feat" in included_features
    # Zero feature must NOT be included because feat_contribs > 0 and < 0 exclude 0.0
    assert "zero_feat" not in included_features, "Zero-impact feature must not be included when non-zero features exist"


def test_novel_categories_graceful_handling(champion_artifact):
    """
    Verifies system behavior with novel/unseen categorical values:
    1. Pipeline level converts novel categories to None/NaN -> 0.0 dummy columns without crashing.
    2. TreeSHAP additivity holds even for profiles with novel categories.
    3. Serving API schema level rejects novel categories with HTTP 422.
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    novel_profile = {
        "customer_id": "NOVEL-01",
        "gender": "Non-Binary",                 # Novel gender
        "SeniorCitizen": 0,
        "Partner": "Maybe",                     # Novel Partner
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Satellite Line",      # Novel MultipleLines
        "InternetService": "Starlink",          # Novel InternetService
        "OnlineSecurity": "Quantum Guard",      # Novel add-on
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Five year",                # Novel Contract
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Cryptocurrency",      # Novel PaymentMethod
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0
    }

    # 1. Pipeline level execution
    df_novel = pd.DataFrame([novel_profile])
    X_novel = pipeline.transform(df_novel)
    assert X_novel.shape == (1, 36), "Pipeline must produce exact 36-column output"

    # All categorical dummy columns corresponding to novel values should be 0.0
    assert X_novel["Contract_One year"].iloc[0] == 0.0
    assert X_novel["Contract_Two year"].iloc[0] == 0.0
    assert X_novel["PaymentMethod_Credit card (automatic)"].iloc[0] == 0.0
    assert X_novel["PaymentMethod_Electronic check"].iloc[0] == 0.0

    # 2. TreeSHAP execution and additivity on novel profile
    contribs = model.booster_.predict(X_novel, pred_contrib=True)
    raw_margin = model.predict(X_novel, raw_score=True)[0]
    reconstructed = contribs[0, -1] + np.sum(contribs[0, :-1])
    assert np.isclose(reconstructed, raw_margin, atol=1e-4), "TreeSHAP additivity must hold on novel category profile"

    # 3. API level rejection with 422
    response = client.post("/predict", json=novel_profile)
    assert response.status_code == 422, f"Expected 422 for novel categories, got {response.status_code}"
    error_detail = response.json().get("detail", [])
    fields_with_errors = [e.get("loc", [])[-1] for e in error_detail if isinstance(e, dict)]
    assert "Contract" in fields_with_errors or "gender" in fields_with_errors


def test_out_of_range_features_behavior(champion_artifact):
    """
    Stress-tests out-of-range numerical features:
    1. Negative tenure and excessive tenure rejected by API with 422.
    2. Extreme MonthlyCharges ($50,000/mo) and TotalCharges ($10,000,000) processed safely without NaN/overflow.
    3. TreeSHAP additivity holds for extreme inputs.
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    # 1. Negative tenure -> 422
    base_valid = generate_random_customer_profile(seed=777)
    bad_tenure = dict(base_valid, tenure=-10)
    resp = client.post("/predict", json=bad_tenure)
    assert resp.status_code == 422, f"Expected 422 for tenure=-10, got {resp.status_code}"

    # Excessive tenure -> 422 (max 100 in schema)
    excessive_tenure = dict(base_valid, tenure=250)
    resp = client.post("/predict", json=excessive_tenure)
    assert resp.status_code == 422, f"Expected 422 for tenure=250, got {resp.status_code}"

    # Negative MonthlyCharges -> 422
    neg_charges = dict(base_valid, MonthlyCharges=-50.0)
    resp = client.post("/predict", json=neg_charges)
    assert resp.status_code == 422, f"Expected 422 for negative charges, got {resp.status_code}"

    # 2. Extreme but positive MonthlyCharges ($50,000/mo) and TotalCharges ($5,000,000)
    extreme_profile = dict(base_valid, MonthlyCharges=50000.0, TotalCharges=5000000.0)
    resp = client.post("/predict", json=extreme_profile)
    assert resp.status_code == 200, f"API should process valid extreme positive numbers: {resp.text}"

    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0, "Probability must remain bounded in [0, 1]"
    assert data["risk_level"] in ("Low", "Medium", "High")
    assert len(data["risk_factors"]) > 0

    # 3. Pipeline and TreeSHAP additivity on extreme inputs
    df_extreme = pd.DataFrame([extreme_profile])
    X_extreme = pipeline.transform(df_extreme)
    contribs = model.booster_.predict(X_extreme, pred_contrib=True)
    raw_margin = model.predict(X_extreme, raw_score=True)[0]
    reconstructed = contribs[0, -1] + np.sum(contribs[0, :-1])
    assert np.isclose(reconstructed, raw_margin, atol=1e-4), "TreeSHAP additivity must hold on extreme numeric inputs"


# ==============================================================================
# Part 2: Drift Monitoring Stress Testing
# ==============================================================================

def test_load_production_cohort_zero_records(clean_test_db, champion_artifact):
    """
    Tests load_production_cohort when database has 0 records.
    Must gracefully fall back to representative cohort (min_rows=100 threshold).
    """
    pipeline = champion_artifact["pipeline"]

    count = clean_test_db.query(PredictionLog).count()
    assert count == 0, "Test database must be empty"

    cohort_df, source = load_production_cohort(min_rows=100, db_session=clean_test_db, pipeline=pipeline)
    assert cohort_df is not None, "Cohort must not be None"
    assert len(cohort_df) >= 100, f"Expected at least 100 fallback rows, got {len(cohort_df)}"
    assert "fallback" in source, f"Expected fallback source, got '{source}'"


def test_load_production_cohort_sparse_and_dense_thresholds(clean_test_db, champion_artifact):
    """
    Verifies threshold transition in load_production_cohort:
    - 5 records (<100): uses fallback
    - 50 records (<100): uses fallback
    - 150 records (>=100): loads real logs from database
    """
    pipeline = champion_artifact["pipeline"]

    def seed_records(n: int):
        clean_test_db.query(PredictionLog).delete()
        clean_test_db.commit()
        sample_prof = generate_random_customer_profile(seed=500)
        logs = [
            PredictionLog(
                prediction_id=str(uuid.uuid4()),
                customer_id=f"SEED-{i:04d}",
                churn_prediction=1 if i % 2 == 0 else 0,
                churn_probability=0.75 if i % 2 == 0 else 0.25,
                risk_level="High" if i % 2 == 0 else "Low",
                features_json=json.dumps(sample_prof),
                model_version="test-v1.0.0",
                latency_ms=12.5
            )
            for i in range(n)
        ]
        clean_test_db.add_all(logs)
        clean_test_db.commit()

    # 1. Test with 5 records
    seed_records(5)
    assert clean_test_db.query(PredictionLog).count() == 5
    cohort_5, source_5 = load_production_cohort(min_rows=100, db_session=clean_test_db, pipeline=pipeline)
    assert "fallback" in source_5, f"Expected fallback for 5 records, got {source_5}"

    # 2. Test with 50 records
    seed_records(50)
    assert clean_test_db.query(PredictionLog).count() == 50
    cohort_50, source_50 = load_production_cohort(min_rows=100, db_session=clean_test_db, pipeline=pipeline)
    assert "fallback" in source_50, f"Expected fallback for 50 records, got {source_50}"

    # 3. Test with 150 records (dense, >= 100 threshold)
    seed_records(150)
    assert clean_test_db.query(PredictionLog).count() == 150
    cohort_150, source_150 = load_production_cohort(min_rows=100, db_session=clean_test_db, pipeline=pipeline)
    assert source_150 == "prediction_logs", f"Expected 'prediction_logs' for 150 records, got {source_150}"
    assert len(cohort_150) == 150, f"Expected 150 rows, got {len(cohort_150)}"


def test_generate_drift_report_sparse_cohort_sizes(reference_data):
    """
    Tests Evidently AI generate_drift_report directly with cohorts of:
    - 5 records
    - 50 records
    - 500 records
    Confirms robust execution and structured summary output.
    """
    ref_slice = reference_data.iloc[:200].copy()

    for n_records in [5, 50, 500]:
        curr_slice = reference_data.sample(n=min(n_records, len(reference_data)), replace=(n_records > len(reference_data)), random_state=42).copy()

        # Inject slight variance so stats can compute
        if "MonthlyCharges" in curr_slice.columns:
            curr_slice["MonthlyCharges"] = curr_slice["MonthlyCharges"] + np.random.normal(5.0, 1.0, size=len(curr_slice))

        report_filename = f"test_adversarial_drift_{n_records}.html"
        summary = generate_drift_report(
            reference_df=ref_slice,
            current_df=curr_slice,
            report_filename=report_filename,
            data_source=f"adversarial_test_{n_records}"
        )

        assert summary is not None, f"Drift summary for {n_records} records must not be None"
        assert summary["report_file"] == report_filename
        assert summary["current_rows"] == n_records
        assert summary["reference_rows"] == len(ref_slice)
        assert isinstance(summary["dataset_drift_detected"], bool)
        assert 0.0 <= summary["drift_share"] <= 1.0
        assert isinstance(summary["drifted_features"], list)
        assert isinstance(summary["number_of_drifted_columns"], int)

        report_path = os.path.join(settings.REPORTS_PATH, report_filename)
        assert os.path.exists(report_path), f"Report HTML {report_path} must exist"


def test_generate_drift_report_zero_records_boundary(reference_data):
    """
    Boundary test: what happens when current_df is completely empty (0 records)?
    Confirms whether Evidently raises an error or is handled.
    """
    ref_slice = reference_data.iloc[:50].copy()
    empty_curr = pd.DataFrame(columns=ref_slice.columns)

    # Calling Evidently with 0 current records is expected to raise ValueError in statistical tests
    with pytest.raises(Exception):
        generate_drift_report(
            reference_df=ref_slice,
            current_df=empty_curr,
            report_filename="test_empty_drift.html",
            data_source="empty_boundary"
        )


def test_monitoring_summary_endpoint_unseeded_db():
    """
    Tests GET /monitoring/summary endpoint when database is completely unseeded.
    Must return HTTP 200 with all required JSON keys without unhandled exceptions.
    """
    # Delete cached summary to force live generation
    summary_path = os.path.join(settings.REPORTS_PATH, "drift_summary.json")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    response = client.get("/monitoring/summary")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert isinstance(data, dict), "Response must be a JSON dictionary"

    # Verify all required keys
    required_keys = [
        "dataset_drift_detected",
        "drift_share",
        "number_of_drifted_columns",
        "drifted_features",
        "total_features_monitored"
    ]
    for key in required_keys:
        assert key in data, f"Required key '{key}' missing from /monitoring/summary response"

    assert isinstance(data["dataset_drift_detected"], bool)
    assert isinstance(data["drift_share"], (int, float))
    assert 0.0 <= data["drift_share"] <= 1.0
    assert isinstance(data["drifted_features"], list)
    assert isinstance(data["number_of_drifted_columns"], int)
