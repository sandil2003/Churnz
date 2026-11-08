"""
Unit tests for TelcoFeaturePipeline.
Validates single-row dummy preservation, batch equivalence, missing value handling,
unseen categories, and target variable extraction.
"""

import numpy as np
import pandas as pd
import pytest
from src.pipeline import TelcoFeaturePipeline


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Fixture providing a diverse 4-row dataset reflecting all main categorical options."""
    return pd.DataFrame([
        {
            "customerID": "CUST-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85",
            "Churn": "No"
        },
        {
            "customerID": "CUST-002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": "1889.5",
            "Churn": "No"
        },
        {
            "customerID": "CUST-003",
            "gender": "Male",
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
            "MonthlyCharges": 95.5,
            "TotalCharges": "191.0",
            "Churn": "Yes"
        },
        {
            "customerID": "CUST-004",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 45,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 20.0,
            "TotalCharges": "900.0",
            "Churn": "No"
        }
    ])


def test_fit_learns_state(sample_training_data):
    """Verifies pipeline learns feature schema, category mapping, and numerical medians."""
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    assert pipeline.is_fitted_ is True
    assert len(pipeline.feature_columns) > 0
    assert pipeline.feature_columns == pipeline.feature_names_
    assert "gender" in pipeline.categories_mapping
    assert sorted(pipeline.categories_mapping["gender"]) == ["Female", "Male"]
    assert "tenure" in pipeline.numerical_medians
    assert pipeline.numerical_medians["tenure"] > 0


def test_single_row_dummy_preservation(sample_training_data):
    """
    Core regression test: verify single customer input retains categorical dummy indicators
    instead of dropping all 26 dummies to 0.0.
    """
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    # Customer 3: Male, Fiber optic, Electronic check, StreamingTV=Yes
    single_customer = pd.DataFrame([{
        "gender": "Male",
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
        "MonthlyCharges": 95.5,
        "TotalCharges": 191.0
    }])

    X_single = pipeline.transform(single_customer)

    assert len(X_single) == 1
    assert list(X_single.columns) == pipeline.feature_columns

    # Verify dummy values are preserved and not zeroed out
    assert X_single["gender_Male"].iloc[0] == 1.0
    assert X_single["InternetService_Fiber optic"].iloc[0] == 1.0
    assert X_single["PaymentMethod_Electronic check"].iloc[0] == 1.0
    assert X_single["StreamingTV_Yes"].iloc[0] == 1.0
    assert X_single["PaperlessBilling_Yes"].iloc[0] == 1.0
    assert X_single["has_fiber_optic"].iloc[0] == 1.0


def test_batch_matches_single_row_exact(sample_training_data):
    """
    Verifies that transforming records in a batch yields the exact same numerical feature
    values as transforming each row individually.
    """
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    X_batch = pipeline.transform(sample_training_data)

    for i in range(len(sample_training_data)):
        single_row = sample_training_data.iloc[[i]]
        X_single = pipeline.transform(single_row)
        np.testing.assert_allclose(
            X_single.values,
            X_batch.iloc[[i]].values,
            err_msg=f"Discrepancy detected at row index {i} between batch and single-row transform!"
        )


def test_missing_and_whitespace_total_charges(sample_training_data):
    """
    Verifies that Kaggle empty strings ' ', empty values, and None in TotalCharges
    are properly imputed without raising exceptions.
    """
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    test_records = pd.DataFrame([
        # Case A: new customer with tenure 0 and whitespace TotalCharges
        {
            "tenure": 0,
            "MonthlyCharges": 25.0,
            "TotalCharges": "   ",
            "gender": "Female"
        },
        # Case B: existing customer with tenure 4 and missing TotalCharges (should impute 4 * 50 = 200)
        {
            "tenure": 4,
            "MonthlyCharges": 50.0,
            "TotalCharges": None,
            "gender": "Male"
        }
    ])

    X = pipeline.transform(test_records)
    assert X["TotalCharges"].iloc[0] == 0.0
    assert X["avg_monthly_charges"].iloc[0] == 0.0
    assert X["TotalCharges"].iloc[1] == 200.0


def test_novel_unseen_categories(sample_training_data):
    """
    Verifies that novel or unseen categories at inference time do not raise errors
    and produce 0.0 across all dummy columns for that feature.
    """
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    novel_customer = pd.DataFrame([{
        "gender": "Non-binary",               # Unseen gender
        "InternetService": "Satellite Starlink", # Unseen internet service
        "PaymentMethod": "Bitcoin Crypto",     # Unseen payment method
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])

    X = pipeline.transform(novel_customer)
    assert len(X) == 1
    # Check that dummy indicators for unseen categories evaluate to 0.0
    assert X["gender_Male"].iloc[0] == 0.0
    assert X["InternetService_Fiber optic"].iloc[0] == 0.0
    assert X["InternetService_No"].iloc[0] == 0.0
    if "PaymentMethod_Electronic check" in X.columns:
        assert X["PaymentMethod_Electronic check"].iloc[0] == 0.0


def test_missing_categorical_column_graceful(sample_training_data):
    """
    Verifies that if an optional categorical column is omitted entirely from the input dataframe,
    the pipeline generates 0.0 for those dummy columns without throwing KeyError or AttributeError.
    """
    pipeline = TelcoFeaturePipeline()
    pipeline.fit(sample_training_data)

    # Input missing InternetService and PaymentMethod entirely
    partial_customer = pd.DataFrame([{
        "gender": "Female",
        "tenure": 10,
        "MonthlyCharges": 40.0
    }])

    X = pipeline.transform(partial_customer)
    assert len(X) == 1
    assert list(X.columns) == pipeline.feature_columns
    assert X["InternetService_Fiber optic"].iloc[0] == 0.0
    assert X["has_fiber_optic"].iloc[0] == 0.0


def test_target_variable_extraction_and_absence(sample_training_data):
    """
    Verifies fit_transform extracts target 'Churn' as a binary series and ensures
    'Churn' is never included in the transformed feature matrix X.
    """
    pipeline = TelcoFeaturePipeline()
    X, y = pipeline.fit_transform(sample_training_data)

    assert y is not None
    assert isinstance(y, pd.Series)
    assert list(y) == [0, 0, 1, 0]
    assert "Churn" not in X.columns
    assert "churn" not in X.columns
    assert "customerID" not in X.columns

    # During transform on data with Churn, Churn must still be excluded from X
    X_trans = pipeline.transform(sample_training_data)
    assert "Churn" not in X_trans.columns


def test_transform_before_fit_raises_error():
    """Verifies that invoking transform() before fit() raises a clear ValueError."""
    pipeline = TelcoFeaturePipeline()
    with pytest.raises(ValueError, match="not fitted yet"):
        pipeline.transform(pd.DataFrame([{"tenure": 5}]))
