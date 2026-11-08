import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.api.predict import compute_treeshap_contributions, format_feature_explanation
from src.schemas import FeatureContribution

client = TestClient(app)


@pytest.fixture(scope="module")
def champion_artifact():
    """Loads champion model artifact directly."""
    return joblib.load(settings.MODEL_PATH)


@pytest.fixture(scope="module")
def sample_raw_data():
    """Loads a slice of raw customer data."""
    raw_df = pd.read_csv(settings.DATA_RAW_PATH)
    if "Churn" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Churn"])
    return raw_df.iloc[:15]


def test_treeshap_output_shape(champion_artifact, sample_raw_data):
    """Verifies TreeSHAP returns matrix of shape (N, num_features + 1)."""
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    X = pipeline.transform(sample_raw_data)
    assert hasattr(model, "booster_"), "Champion model must contain native LightGBM booster"

    contribs = model.booster_.predict(X, pred_contrib=True)
    num_features = len(pipeline.feature_columns)
    assert contribs.shape == (len(sample_raw_data), num_features + 1)


def test_treeshap_exact_sum_to_margin(champion_artifact, sample_raw_data):
    """
    Mathematical verification of TreeSHAP efficiency property:
    sum(feature_attributions) + base_value == model_raw_score (margin/log-odds).
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    X = pipeline.transform(sample_raw_data)
    contribs = model.booster_.predict(X, pred_contrib=True)
    raw_scores = model.predict(X, raw_score=True)

    contrib_sums = np.sum(contribs, axis=1)
    np.testing.assert_allclose(
        contrib_sums,
        raw_scores,
        atol=1e-4,
        err_msg="TreeSHAP contributions must sum exactly to raw margin score"
    )


def test_treeshap_sigmoid_matches_predicted_probability(champion_artifact, sample_raw_data):
    """
    Mathematical verification that sigmoid of margin score matches predict_proba:
    probability = 1 / (1 + exp(-margin))
    """
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    X = pipeline.transform(sample_raw_data)
    raw_scores = model.predict(X, raw_score=True)
    probs = model.predict_proba(X)[:, 1]

    sigmoid_probs = 1.0 / (1.0 + np.exp(-raw_scores))
    np.testing.assert_allclose(
        sigmoid_probs,
        probs,
        atol=1e-4,
        err_msg="Sigmoid of margin score must match predict_proba"
    )


def test_treeshap_base_value_invariance(champion_artifact, sample_raw_data):
    """Verifies that the TreeSHAP expected value (base bias) is constant across all inputs."""
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    X = pipeline.transform(sample_raw_data)
    contribs = model.booster_.predict(X, pred_contrib=True)
    base_values = contribs[:, -1]

    assert np.allclose(base_values, base_values[0]), "Base value must be constant across all records"


def test_compute_treeshap_ranking_and_direction(champion_artifact, sample_raw_data):
    """Verifies that compute_treeshap_contributions accurately segregates risk and protective features."""
    model = champion_artifact["model"]
    pipeline = champion_artifact["pipeline"]

    row = sample_raw_data.iloc[0:1]
    X_row = pipeline.transform(row)
    raw_dict = row.iloc[0].to_dict()

    risk_factors, explanations = compute_treeshap_contributions(
        model=model,
        pipeline=pipeline,
        X=X_row,
        raw_record=raw_dict,
        top_pos=3,
        top_neg=2
    )

    assert len(risk_factors) > 0
    assert len(explanations) == len(risk_factors)

    for factor in risk_factors:
        assert isinstance(factor, FeatureContribution)
        if factor.direction == "increases_risk":
            assert factor.attribution > 0, "Risk-increasing features must have positive attributions"
        elif factor.direction == "decreases_risk":
            assert factor.attribution < 0, "Risk-decreasing features must have negative attributions"
        assert len(factor.display_text) > 0


def test_predict_api_structured_treeshap_contract():
    """End-to-end integration test validating TreeSHAP attribution output in /predict API."""
    customer = {
        "customer_id": "TEST-SHAP-01",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
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
        "MonthlyCharges": 99.85,
        "TotalCharges": 199.70
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    res = response.json()

    assert "risk_factors" in res
    assert len(res["risk_factors"]) >= 2

    # Check that highest positive risk contributors are captured
    features = [rf["feature"] for rf in res["risk_factors"]]
    # High-risk profile should feature month-to-month or tenure or payment method
    assert any(f in features for f in ["is_month_to_month", "tenure", "PaymentMethod_Electronic check", "has_fiber_optic"])


def test_compute_treeshap_graceful_fallback_mock():
    """Verifies that non-booster models fall back cleanly without unhandled crashes."""
    class DummyModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.5, -0.3, 0.1])

    class DummyPipeline:
        feature_columns = ["feat_a", "feat_b", "feat_c"]

    dummy_model = DummyModel()
    dummy_pipe = DummyPipeline()
    X_dummy = np.zeros((1, 3))

    risk_factors, explanations = compute_treeshap_contributions(
        model=dummy_model,
        pipeline=dummy_pipe,
        X=X_dummy
    )

    assert len(risk_factors) > 0
    assert len(explanations) == len(risk_factors)
