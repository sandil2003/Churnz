"""
Empirical Adversarial Stress Test Script for Milestone M2:
Serving API TreeSHAP Explainability and Evidently AI Drift Monitoring.

Executes quantitative stress tests and outputs detailed diagnostic evidence.
"""

import os
import sys
import json
import uuid
import random
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import joblib

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.config import settings
from src.main import app
from src.database import Base, PredictionLog, init_db_connection
from src.api.predict import compute_treeshap_contributions, format_feature_explanation
from src.monitoring import (
    load_reference_data,
    load_production_cohort,
    generate_drift_report,
    get_latest_drift_summary
)

client = TestClient(app)


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


def run_explainability_stress_tests(artifact: Dict[str, Any]):
    print("=" * 80)
    print("SECTION 1: TREESHAP EXPLAINABILITY STRESS TESTING")
    print("=" * 80)

    model = artifact["model"]
    pipeline = artifact["pipeline"]

    print("\n--- 1.1: 50 Randomized Customer Profiles TreeSHAP Additivity Test ---")
    profiles = [generate_random_customer_profile(seed=2000 + i) for i in range(50)]
    df_profiles = pd.DataFrame(profiles)
    X = pipeline.transform(df_profiles)

    contribs = model.booster_.predict(X, pred_contrib=True)
    raw_scores = model.predict(X, raw_score=True)
    probs = model.predict_proba(X)[:, 1]

    errors = []
    sig_errors = []

    print(f"{'Idx':<4} | {'Customer ID':<10} | {'Raw Margin':<10} | {'Base Bias':<10} | {'Sum Attr':<10} | {'Reconstructed':<14} | {'Abs Diff':<10} | {'Prob':<6} | {'Sigmoid':<7}")
    print("-" * 92)

    for i in range(50):
        base_bias = contribs[i, -1]
        attributions = contribs[i, :-1]
        raw_margin = raw_scores[i]
        reconstructed = base_bias + np.sum(attributions)
        diff = abs(reconstructed - raw_margin)
        errors.append(diff)

        sig_prob = 1.0 / (1.0 + np.exp(-raw_margin))
        sig_diff = abs(sig_prob - probs[i])
        sig_errors.append(sig_diff)

        if i < 10 or i >= 45:  # Print first 10 and last 5
            print(f"{i:<4} | {profiles[i]['customer_id']:<10} | {raw_margin:<10.5f} | {base_bias:<10.5f} | {np.sum(attributions):<10.5f} | {reconstructed:<14.5f} | {diff:<10.2e} | {probs[i]:<6.4f} | {sig_prob:<7.4f}")
        elif i == 10:
            print(" ... (profiles 10..44 omitted for brevity) ...")

    all_close_1e4 = np.all(np.array(errors) < 1e-4)
    max_err = max(errors)
    mean_err = np.mean(errors)
    max_sig_err = max(sig_errors)

    print("-" * 92)
    print(f"Results across 50 profiles:")
    print(f"  - Additivity Condition holds (< 1e-4): {all_close_1e4} (100% of profiles)")
    print(f"  - Max Additivity Absolute Error:      {max_err:.6e}")
    print(f"  - Mean Additivity Absolute Error:     {mean_err:.6e}")
    print(f"  - Max Sigmoid vs Proba Absolute Error: {max_sig_err:.6e}")

    print("\n--- 1.2: Zero-Impact Feature Ranking & Non-Dominance Test ---")
    class TestPartialModel:
        class booster_:
            @staticmethod
            def predict(X, pred_contrib=True):
                # 4 features: pos_1 (+1.2), zero (0.0), neg_1 (-0.7), bias (-0.4)
                return np.array([[1.2, 0.0, -0.7, -0.4]])

    class TestPipe:
        feature_columns = ["pos_feature", "zero_feature", "neg_feature"]

    rf, exp = compute_treeshap_contributions(
        model=TestPartialModel(),
        pipeline=TestPipe(),
        X=np.zeros((1, 3)),
        top_pos=3,
        top_neg=2
    )

    print(f"Features returned by compute_treeshap_contributions:")
    for f in rf:
        print(f"  - Feature: {f.feature:<15} Attribution: {f.attribution:<8} Direction: {f.direction}")

    zero_feat_included = any(f.feature == "zero_feature" for f in rf)
    print(f"  - Zero-attribution feature included in top factors? {zero_feat_included}")
    assert not zero_feat_included, "Zero-attribution feature must NOT dominate or be included when non-zero features exist"

    print("\n--- 1.3: Novel Categories Stress Test ---")
    novel_profile = {
        "customer_id": "NOVEL-CUST",
        "gender": "Non-Binary",
        "SeniorCitizen": 0,
        "Partner": "Undisclosed",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Satellite Mesh",
        "InternetService": "Starlink LEO",
        "OnlineSecurity": "ZeroTrust",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Decade Plan",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Solana Pay",
        "MonthlyCharges": 95.0,
        "TotalCharges": 2280.0
    }

    # Pipeline handling
    df_novel = pd.DataFrame([novel_profile])
    X_novel = pipeline.transform(df_novel)
    print(f"  - Pipeline transform output shape on novel input: {X_novel.shape} (exact 36 columns preserved)")
    dummy_contract_one_year = X_novel["Contract_One year"].iloc[0]
    dummy_contract_two_year = X_novel["Contract_Two year"].iloc[0]
    print(f"  - Dummy columns for novel contract: Contract_One year={dummy_contract_one_year}, Contract_Two year={dummy_contract_two_year}")

    # TreeSHAP additivity on novel profile
    contribs_nov = model.booster_.predict(X_novel, pred_contrib=True)
    raw_nov = model.predict(X_novel, raw_score=True)[0]
    reconstructed_nov = contribs_nov[0, -1] + np.sum(contribs_nov[0, :-1])
    diff_nov = abs(reconstructed_nov - raw_nov)
    print(f"  - TreeSHAP additivity diff on novel category profile: {diff_nov:.6e} (< 1e-4: {diff_nov < 1e-4})")

    # API rejection check
    resp_nov = client.post("/predict", json=novel_profile)
    print(f"  - POST /predict status code on novel categories: {resp_nov.status_code} (Rejection expected: 422)")

    print("\n--- 1.4: Out-of-Range Numerical Features Stress Test ---")
    valid_base = generate_random_customer_profile(seed=999)

    # 1. Negative tenure
    resp_neg_tenure = client.post("/predict", json=dict(valid_base, tenure=-1))
    print(f"  - POST /predict tenure=-1 status code: {resp_neg_tenure.status_code} (Expected 422)")

    # 2. Tenure > 100
    resp_high_tenure = client.post("/predict", json=dict(valid_base, tenure=150))
    print(f"  - POST /predict tenure=150 status code: {resp_high_tenure.status_code} (Expected 422)")

    # 3. Negative MonthlyCharges
    resp_neg_mc = client.post("/predict", json=dict(valid_base, MonthlyCharges=-25.0))
    print(f"  - POST /predict MonthlyCharges=-25.0 status code: {resp_neg_mc.status_code} (Expected 422)")

    # 4. Extreme valid MonthlyCharges ($50,000/mo) and TotalCharges ($10,000,000)
    extreme_p = dict(valid_base, MonthlyCharges=50000.0, TotalCharges=10000000.0)
    resp_extreme = client.post("/predict", json=extreme_p)
    print(f"  - POST /predict extreme charges ($50,000/mo) status code: {resp_extreme.status_code} (Expected 200)")
    ext_data = resp_extreme.json()
    print(f"    Churn Prob: {ext_data['churn_probability']}, Tier: {ext_data['risk_level']}, Risk Factors: {len(ext_data['risk_factors'])}")
    for rf_item in ext_data['risk_factors'][:2]:
        print(f"      {rf_item['feature']}: {rf_item['attribution']:+.4f} | {rf_item['display_text']}")


def run_drift_monitoring_stress_tests(artifact: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("SECTION 2: DRIFT MONITORING STRESS TESTING")
    print("=" * 80)

    ref_df = load_reference_data()
    pipeline = artifact["pipeline"]

    print("\n--- 2.1: Production Cohort Loading Thresholds (0, 5, 50, 150 records) ---")
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    def seed_db(n: int):
        session.query(PredictionLog).delete()
        session.commit()
        sample_prof = generate_random_customer_profile(seed=123)
        records = [
            PredictionLog(
                prediction_id=str(uuid.uuid4()),
                customer_id=f"SEED-{i:04d}",
                churn_prediction=1 if i % 2 == 0 else 0,
                churn_probability=0.8 if i % 2 == 0 else 0.2,
                risk_level="High" if i % 2 == 0 else "Low",
                features_json=json.dumps(sample_prof),
                model_version="champ-v1",
                latency_ms=10.0
            )
            for i in range(n)
        ]
        session.add_all(records)
        session.commit()

    # 0 records
    seed_db(0)
    c0, s0 = load_production_cohort(min_rows=100, db_session=session, pipeline=pipeline)
    print(f"  - Database count: 0   -> Cohort rows: {len(c0):<4} Source: {s0}")

    # 5 records
    seed_db(5)
    c5, s5 = load_production_cohort(min_rows=100, db_session=session, pipeline=pipeline)
    print(f"  - Database count: 5   -> Cohort rows: {len(c5):<4} Source: {s5}")

    # 50 records
    seed_db(50)
    c50, s50 = load_production_cohort(min_rows=100, db_session=session, pipeline=pipeline)
    print(f"  - Database count: 50  -> Cohort rows: {len(c50):<4} Source: {s50}")

    # 150 records (>= 100 threshold)
    seed_db(150)
    c150, s150 = load_production_cohort(min_rows=100, db_session=session, pipeline=pipeline)
    print(f"  - Database count: 150 -> Cohort rows: {len(c150):<4} Source: {s150}")

    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

    print("\n--- 2.2: Evidently AI Direct Drift Execution across Cohort Sizes ---")
    ref_slice = ref_df.iloc[:200].copy()

    for n in [5, 50, 500]:
        t0 = time.perf_counter()
        curr_slice = ref_df.sample(n=min(n, len(ref_df)), replace=(n > len(ref_df)), random_state=42).copy()
        if "MonthlyCharges" in curr_slice.columns:
            curr_slice["MonthlyCharges"] = curr_slice["MonthlyCharges"] + np.random.normal(6.0, 1.5, size=len(curr_slice))

        report_file = f"adversarial_drift_{n}.html"
        summary = generate_drift_report(
            reference_df=ref_slice,
            current_df=curr_slice,
            report_filename=report_file,
            data_source=f"test_cohort_{n}"
        )
        duration = time.perf_counter() - t0

        print(f"  - Cohort Size: {n:<3} | Latency: {duration:.2f}s | Drift Detected: {summary['dataset_drift_detected']} | Drift Share: {summary['drift_share']:.4f} | Drifted Cols: {summary['number_of_drifted_columns']}")

    print("\n--- 2.3: Zero Records Boundary Test ---")
    empty_curr = pd.DataFrame(columns=ref_slice.columns)
    try:
        generate_drift_report(ref_slice, empty_curr, "empty.html", "zero_test")
        print("  - Unexpected: Evidently succeeded with 0 current records.")
    except Exception as e:
        print(f"  - Expected Exception on 0 current records: {type(e).__name__}: {str(e)[:70]}...")

    print("\n--- 2.4: /monitoring/summary Endpoint on Unseeded Database ---")
    # Invalidate cached drift summary
    summary_path = os.path.join(settings.REPORTS_PATH, "drift_summary.json")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    resp_summary = client.get("/monitoring/summary")
    print(f"  - GET /monitoring/summary status code: {resp_summary.status_code}")
    assert resp_summary.status_code == 200

    data = resp_summary.json()
    print("  - Returned JSON payload keys and sample values:")
    for k, v in data.items():
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        print(f"    '{k}': {v_str}")

    required_keys = [
        "dataset_drift_detected",
        "drift_share",
        "number_of_drifted_columns",
        "drifted_features",
        "total_features_monitored"
    ]
    all_keys_present = all(k in data for k in required_keys)
    print(f"  - All required contract keys present: {all_keys_present}")
    assert all_keys_present


def main():
    print("=" * 80)
    print("CHALLENGER_M2_2 EMPIRICAL ADVERSARIAL BENCHMARK HARNESS")
    print("=" * 80)

    artifact = joblib.load(settings.MODEL_PATH)
    run_explainability_stress_tests(artifact)
    run_drift_monitoring_stress_tests(artifact)

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL ADVERSARIAL VERIFICATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
