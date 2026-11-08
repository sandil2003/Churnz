"""
Adversarial stress testing and empirical verification suite for TelcoFeaturePipeline.
Tests single-row vs batch inference parity, unseen categoricals, missing/dirty values,
tenure=0 boundary conditions, schema determinism, and input immutability.
"""

import copy
import sys
import os
import numpy as np
import pandas as pd
import pytest
import joblib

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.pipeline import TelcoFeaturePipeline


@pytest.fixture(scope="module")
def champion_artifact():
    """Loads champion model bundle containing the production fitted pipeline."""
    model_path = os.path.join(backend_dir, "models", "champion_model.joblib")
    assert os.path.exists(model_path), f"Champion model not found at {model_path}"
    bundle = joblib.load(model_path)
    return bundle


@pytest.fixture(scope="module")
def raw_dataset():
    """Loads raw Telco customer churn dataset."""
    data_path = os.path.join(backend_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert os.path.exists(data_path), f"Raw data not found at {data_path}"
    return pd.read_csv(data_path)


def test_batch_vs_single_parity_100_rows(champion_artifact, raw_dataset):
    """
    STRESS TEST 1: Batch vs Single-Row Equivalence & Parity.
    Tests transforming each of 100 diverse customers individually vs transforming
    them all together in a single batch of 100.
    Verifies bitwise / numeric parity of feature values and LightGBM output probabilities.
    """
    pipeline = champion_artifact["pipeline"]
    model = champion_artifact["model"]

    # Select 100 rows with diverse indices
    sample_100 = raw_dataset.iloc[:100].copy()

    # Batch transform
    X_batch = pipeline.transform(sample_100)
    probs_batch = model.predict_proba(X_batch)[:, 1]

    # Verify shape
    assert X_batch.shape == (100, 36), f"Expected shape (100, 36), got {X_batch.shape}"

    # Individual single-row transforms
    for i in range(100):
        single_row = sample_100.iloc[[i]].copy()
        X_single = pipeline.transform(single_row)
        prob_single = model.predict_proba(X_single)[0, 1]

        # Feature matrix comparison
        np.testing.assert_allclose(
            X_single.values,
            X_batch.iloc[[i]].values,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Discrepancy in feature matrix at index {i} ({single_row['customerID'].values[0]})"
        )

        # LightGBM inference probability comparison
        np.testing.assert_allclose(
            prob_single,
            probs_batch[i],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Discrepancy in LightGBM predicted probability at index {i}"
        )


def test_batch_context_invariance(champion_artifact, raw_dataset):
    """
    STRESS TEST 2: Context Invariance.
    Verifies that customer A produces the exact same transformed vector regardless of
    what other customers are in the batch (empty rows, duplicate rows, diametrically opposite customers).
    """
    pipeline = champion_artifact["pipeline"]

    target_customer = raw_dataset.iloc[[15]].copy()  # Specific customer
    X_isolated = pipeline.transform(target_customer)

    # Context 1: Target customer with 10 random other customers
    batch_random = pd.concat([target_customer, raw_dataset.iloc[20:30]], ignore_index=True)
    X_in_random = pipeline.transform(batch_random)
    np.testing.assert_allclose(X_isolated.values, X_in_random.iloc[[0]].values, atol=1e-12)

    # Context 2: Target customer placed in the middle (index 5)
    batch_middle = pd.concat([raw_dataset.iloc[0:5], target_customer, raw_dataset.iloc[5:10]], ignore_index=True)
    X_in_middle = pipeline.transform(batch_middle)
    np.testing.assert_allclose(X_isolated.values, X_in_middle.iloc[[5]].values, atol=1e-12)

    # Context 3: Target customer placed at the end
    batch_end = pd.concat([raw_dataset.iloc[40:50], target_customer], ignore_index=True)
    X_in_end = pipeline.transform(batch_end)
    np.testing.assert_allclose(X_isolated.values, X_in_end.iloc[[10]].values, atol=1e-12)

    # Context 4: Target customer with 10 duplicate copies of themselves
    batch_duplicates = pd.concat([target_customer] * 11, ignore_index=True)
    X_duplicates = pipeline.transform(batch_duplicates)
    for k in range(11):
        np.testing.assert_allclose(X_isolated.values, X_duplicates.iloc[[k]].values, atol=1e-12)

    # Context 5: Target customer with 10 empty/NaN customer rows
    empty_rows = pd.DataFrame([{}] * 10)
    batch_with_empty = pd.concat([target_customer, empty_rows], ignore_index=True)
    X_with_empty = pipeline.transform(batch_with_empty)
    np.testing.assert_allclose(X_isolated.values, X_with_empty.iloc[[0]].values, atol=1e-12)


def test_adversarial_novel_and_unseen_categories(champion_artifact):
    """
    STRESS TEST 3: Novel, Unseen, and Out-of-Vocabulary Categorical Values.
    Verifies that unknown categorical levels do not cause crashes, NaNs, or unexpected dummy columns.
    All dummy columns for that categorical variable must cleanly evaluate to 0.0.
    """
    pipeline = champion_artifact["pipeline"]
    model = champion_artifact["model"]

    adversarial_inputs = [
        # Customer 1: Unseen contract, internet service, payment method, gender
        {
            "gender": "Non-binary",
            "Contract": "5-year",
            "InternetService": "Satellite Starlink",
            "PaymentMethod": "Cryptocurrency Bitcoin",
            "OnlineSecurity": "Quantum Encrypted",
            "tenure": 10,
            "MonthlyCharges": 120.0,
            "TotalCharges": 1200.0
        },
        # Customer 2: Mixed casing and spaces
        {
            "gender": "female",  # lowercase
            "Contract": "two year",  # lowercase
            "InternetService": "fiber optic",  # lowercase
            "PaymentMethod": "electronic check",  # lowercase
            "tenure": 24,
            "MonthlyCharges": 80.0,
            "TotalCharges": 1920.0
        },
        # Customer 3: Non-string data types in categoricals
        {
            "gender": 12345,
            "Contract": True,
            "PaymentMethod": 99.99,
            "PhoneService": None,
            "MultipleLines": False,
            "tenure": 6,
            "MonthlyCharges": 45.0,
            "TotalCharges": 270.0
        },
        # Customer 4: Emojis and adversarial injection strings
        {
            "gender": "👩",
            "Contract": "'; DROP TABLE customers; --",
            "InternetService": "<script>alert('xss')</script>",
            "PaymentMethod": "$$$ CASH $$$",
            "tenure": 12,
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0
        }
    ]

    df_adv = pd.DataFrame(adversarial_inputs)
    X_adv = pipeline.transform(df_adv)

    assert X_adv.shape == (4, 36)
    assert not X_adv.isna().any().any(), "Transformed adversarial features contain NaN values!"
    assert not np.isinf(X_adv.values).any(), "Transformed adversarial features contain Inf values!"

    # In Customer 1, check that all dummy columns for unseen features are 0.0
    c1 = X_adv.iloc[0]
    assert c1["gender_Male"] == 0.0
    assert c1["Contract_One year"] == 0.0
    assert c1["Contract_Two year"] == 0.0
    assert c1["InternetService_Fiber optic"] == 0.0
    assert c1["InternetService_No"] == 0.0
    assert c1["PaymentMethod_Credit card (automatic)"] == 0.0
    assert c1["PaymentMethod_Electronic check"] == 0.0
    assert c1["PaymentMethod_Mailed check"] == 0.0

    # Model inference must succeed on all adversarial rows
    probs = model.predict_proba(X_adv)[:, 1]
    assert len(probs) == 4
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_adversarial_empty_strings_nans_and_missing_columns(champion_artifact):
    """
    STRESS TEST 4: Empty Strings, Missing Columns, and NaNs.
    Verifies that partial or completely empty records impute gracefully using learned medians
    without raising KeyError, AttributeError, or generating NaNs.
    """
    pipeline = champion_artifact["pipeline"]
    model = champion_artifact["model"]

    # Case A: Completely empty dictionary
    df_empty_dict = pd.DataFrame([{}])
    X_empty = pipeline.transform(df_empty_dict)
    assert X_empty.shape == (1, 36)
    assert not X_empty.isna().any().any()
    # Check that numericals fell back to learned medians
    assert X_empty["tenure"].iloc[0] == 0.0  # tenure missing defaults to 0 in clean_raw_data
    assert X_empty["MonthlyCharges"].iloc[0] == 0.0
    assert X_empty["TotalCharges"].iloc[0] == 0.0

    # Case B: All categorical columns are empty strings or whitespace
    all_cats_empty = {col: "   " for col in pipeline.categorical_columns}
    all_cats_empty["tenure"] = 5
    all_cats_empty["MonthlyCharges"] = 50.0
    all_cats_empty["TotalCharges"] = 250.0
    X_cats_empty = pipeline.transform(pd.DataFrame([all_cats_empty]))
    assert X_cats_empty.shape == (1, 36)
    assert not X_cats_empty.isna().any().any()

    # Case C: All columns are NaN or None
    all_nan = {col: np.nan for col in pipeline.categorical_columns}
    all_nan.update({col: None for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]})
    X_all_nan = pipeline.transform(pd.DataFrame([all_nan]))
    assert X_all_nan.shape == (1, 36)
    assert not X_all_nan.isna().any().any()

    # Model inference should execute cleanly
    p = model.predict_proba(X_all_nan)[0, 1]
    assert 0.0 <= p <= 1.0


def test_adversarial_tenure_zero_boundary(champion_artifact):
    """
    STRESS TEST 5: Tenure = 0 and TotalCharges Boundary Variations.
    Stress-tests Kaggle edge cases where tenure=0 and TotalCharges is missing, whitespace,
    zero string, float, or None.
    """
    pipeline = champion_artifact["pipeline"]

    test_cases = [
        # 1. tenure=0, TotalCharges=" " (Kaggle classic whitespace)
        {"tenure": 0, "MonthlyCharges": 25.0, "TotalCharges": " "},
        # 2. tenure=0, TotalCharges="   " (multiple spaces)
        {"tenure": 0, "MonthlyCharges": 30.0, "TotalCharges": "   "},
        # 3. tenure=0, TotalCharges="" (empty string)
        {"tenure": 0, "MonthlyCharges": 35.0, "TotalCharges": ""},
        # 4. tenure=0, TotalCharges=0 (integer zero)
        {"tenure": 0, "MonthlyCharges": 40.0, "TotalCharges": 0},
        # 5. tenure=0, TotalCharges=0.0 (float zero)
        {"tenure": 0, "MonthlyCharges": 45.0, "TotalCharges": 0.0},
        # 6. tenure=0, TotalCharges="0" (string zero)
        {"tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": "0"},
        # 7. tenure=0, TotalCharges="0.0" (string float zero)
        {"tenure": 0, "MonthlyCharges": 55.0, "TotalCharges": "0.0"},
        # 8. tenure=0, TotalCharges=None
        {"tenure": 0, "MonthlyCharges": 60.0, "TotalCharges": None},
        # 9. tenure=0, TotalCharges=np.nan
        {"tenure": 0, "MonthlyCharges": 65.0, "TotalCharges": np.nan},
        # 10. tenure=0, TotalCharges missing from dict entirely
        {"tenure": 0, "MonthlyCharges": 70.0},
    ]

    for idx, tc in enumerate(test_cases):
        df_tc = pd.DataFrame([tc])
        X = pipeline.transform(df_tc)
        assert X.shape == (1, 36)
        assert not X.isna().any().any(), f"NaN generated in test case {idx}: {tc}"
        assert X["TotalCharges"].iloc[0] == 0.0, f"Expected TotalCharges=0.0 in case {idx}, got {X['TotalCharges'].iloc[0]}"
        assert X["avg_monthly_charges"].iloc[0] == 0.0, f"Expected avg_monthly_charges=0.0 in case {idx}, got {X['avg_monthly_charges'].iloc[0]}"
        assert X["tenure_year"].iloc[0] == 0.0


def test_schema_determinism_and_column_order(champion_artifact, raw_dataset):
    """
    STRESS TEST 6: Strict Schema Determinism and Column Ordering.
    Verifies that output column names, ordering, and dtypes are strictly identical regardless
    of how the input DataFrame columns are structured or ordered.
    """
    pipeline = champion_artifact["pipeline"]
    expected_columns = pipeline.feature_columns

    sample = raw_dataset.iloc[:5].copy()

    # Transform 1: Normal column order
    X1 = pipeline.transform(sample)

    # Transform 2: Reversed column order
    sample_reversed = sample[list(reversed(sample.columns))].copy()
    X2 = pipeline.transform(sample_reversed)

    # Transform 3: Shuffled column order
    cols_shuffled = list(sample.columns)
    np.random.seed(42)
    np.random.shuffle(cols_shuffled)
    sample_shuffled = sample[cols_shuffled].copy()
    X3 = pipeline.transform(sample_shuffled)

    # Transform 4: Input with extra unrelated columns
    sample_extra = sample.copy()
    sample_extra["EXTRA_METADATA_1"] = "unrelated_value"
    sample_extra["EXTRA_NUMERIC_2"] = 9999.99
    sample_extra["customer_uuid"] = "123e4567-e89b-12d3-a456-426614174000"
    X4 = pipeline.transform(sample_extra)

    # Verifications
    for name, X in [("reversed", X2), ("shuffled", X3), ("extra_cols", X4)]:
        assert list(X.columns) == expected_columns, f"Column order mismatched in {name}"
        assert X.shape == (5, 36), f"Shape mismatch in {name}"
        assert all(dtype == np.float64 for dtype in X.dtypes), f"Dtype not float64 in {name}"
        np.testing.assert_allclose(X1.values, X.values, atol=1e-12, err_msg=f"Values mismatched in {name}")


def test_input_dataframe_immutability(champion_artifact, raw_dataset):
    """
    STRESS TEST 7: Input DataFrame Immutability.
    Verifies that pipeline.transform() never mutates or modifies the caller's DataFrame.
    """
    pipeline = champion_artifact["pipeline"]

    original_df = raw_dataset.iloc[:10].copy()
    original_df_backup = original_df.copy(deep=True)

    _ = pipeline.transform(original_df)

    # Verify input DataFrame was untouched
    pd.testing.assert_frame_equal(original_df, original_df_backup)


def test_extreme_numerical_inputs(champion_artifact):
    """
    STRESS TEST 8: Extreme and Malformed Numerical Values.
    Verifies behavior when tenure or charges contain negative, string junk, or extreme values.
    """
    pipeline = champion_artifact["pipeline"]
    model = champion_artifact["model"]

    extreme_df = pd.DataFrame([
        # Negative tenure
        {"tenure": -5, "MonthlyCharges": 50.0, "TotalCharges": 250.0},
        # String junk in numericals
        {"tenure": "INVALID_TENURE", "MonthlyCharges": "NOT_A_NUMBER", "TotalCharges": "CORRUPTED"},
        # Huge numbers
        {"tenure": 1000, "MonthlyCharges": 999999.0, "TotalCharges": 999999000.0},
    ])

    X = pipeline.transform(extreme_df)
    assert X.shape == (3, 36)
    # Check that string junk became fallback 0 / numerical medians
    row1 = X.iloc[1]
    assert row1["tenure"] == 0.0
    assert row1["MonthlyCharges"] == 0.0
    assert row1["TotalCharges"] == 0.0
    assert not X.isna().any().any()

    # Verify model prediction handles these inputs without crashing
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == 3


if __name__ == "__main__":
    pytest.main(["-v", __file__])
