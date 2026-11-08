import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.config import settings
import src.database as db_mod
from src.database import PredictionLog, init_db_connection

logger = logging.getLogger(__name__)


def load_reference_data() -> Optional[pd.DataFrame]:
    """Loads the baseline reference dataset created during champion model training."""
    ref_path = os.path.join(settings.DATA_PROCESSED_PATH, "reference_data.csv")
    if os.path.exists(ref_path):
        try:
            df = pd.read_csv(ref_path)
            # Drop non-feature columns if present
            feature_cols = [c for c in df.columns if c not in ("Churn", "target", "prediction", "customer_id")]
            return df[feature_cols]
        except Exception as e:
            logger.error(f"Error loading reference dataset from {ref_path}: {e}")
            return None
    return None


def load_production_cohort(
    min_rows: int = 100,
    db_session=None,
    pipeline=None
) -> Tuple[pd.DataFrame, str]:
    """
    Retrieves production feature data for drift comparison.
    Queries PostgreSQL prediction_logs if available and count >= min_rows.
    Otherwise gracefully falls back to a realistic production cohort sampled
    from the raw dataset with feature engineering applied.

    Returns:
        Tuple[pd.DataFrame, str]: (cohort_df, source_description)
    """
    # 1. Attempt to query real logs from PostgreSQL
    active_session = db_session
    created_locally = False

    if active_session is None:
        if db_mod.SessionLocal is None:
            db_mod.init_db_connection()
        if db_mod.SessionLocal is not None:
            try:
                active_session = db_mod.SessionLocal()
                created_locally = True
            except Exception as e:
                logger.warning(f"Could not instantiate database session for drift query: {e}")

    if active_session is not None:
        try:
            count = active_session.query(PredictionLog).count()
            if count >= min_rows:
                logs = (
                    active_session.query(PredictionLog.features_json)
                    .order_by(PredictionLog.created_at.desc())
                    .limit(1000)
                    .all()
                )
                records = [json.loads(row[0]) for row in logs]
                raw_df = pd.DataFrame(records)

                if pipeline is not None:
                    transformed_df = pipeline.transform(raw_df)
                    logger.info(f"Loaded {len(transformed_df)} live prediction records from PostgreSQL.")
                    return transformed_df, "prediction_logs"
                else:
                    return raw_df, "prediction_logs_raw"
        except Exception as e:
            logger.warning(f"Querying prediction_logs failed: {e}. Falling back to baseline cohort.")
        finally:
            if created_locally and active_session is not None:
                active_session.close()

    # 2. Fallback: Generate production cohort from raw data with pipeline transformation
    logger.info("Using representative production cohort fallback (<100 DB logs or DB offline).")
    raw_path = settings.DATA_RAW_PATH
    if os.path.exists(raw_path):
        try:
            raw_full = pd.read_csv(raw_path)
            # Take a sample of 500 records
            sample = raw_full.sample(n=min(500, len(raw_full)), random_state=42).copy()
            if "Churn" in sample.columns:
                sample = sample.drop(columns=["Churn"])

            # Induce slight realistic distribution shifts (e.g. inflation and tenure dynamics)
            if "MonthlyCharges" in sample.columns:
                sample["MonthlyCharges"] = sample["MonthlyCharges"] + np.random.normal(8.0, 2.0, size=len(sample))
            if "tenure" in sample.columns:
                sample["tenure"] = np.maximum(0, sample["tenure"] - 5)

            if pipeline is not None:
                transformed = pipeline.transform(sample)
                return transformed, "fallback_cohort_transformed"
            return sample, "fallback_cohort_raw"
        except Exception as e:
            logger.error(f"Fallback cohort generation from raw data failed: {e}")

    # 3. Ultimate Fallback: Perturb reference data directly
    ref_df = load_reference_data()
    if ref_df is not None:
        sample = ref_df.sample(n=min(200, len(ref_df)), random_state=42).copy()
        if "MonthlyCharges" in sample.columns:
            sample["MonthlyCharges"] = sample["MonthlyCharges"] * 1.08
        return sample, "reference_sample_fallback"

    # Emergency dummy fallback to prevent unhandled crash
    dummy = pd.DataFrame({"MonthlyCharges": [70.0, 85.0], "tenure": [12, 24]})
    return dummy, "emergency_dummy"


def extract_drift_metrics_from_snapshot(snapshot: Any, column_count: int) -> Dict[str, Any]:
    """Extracts high-level dataset drift KPIs and drifted features from Evidently snapshot."""
    drift_detected = False
    drift_share = 0.0
    drifted_cols_count = 0
    drifted_features: List[str] = []

    try:
        data = snapshot.dict()
        for metric in data.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            val = metric.get("value")

            # Extract summary drift counts
            if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
                drifted_cols_count = int(val.get("count", 0))
                drift_share = float(val.get("share", 0.0))

            # Extract per-column drift
            if "ValueDrift" in metric_name:
                cfg = metric.get("config", {})
                col = cfg.get("column")
                threshold = float(cfg.get("threshold", 0.05))
                # For p_value methods, drift is detected when p_value < threshold
                if isinstance(val, (int, float)) and val < threshold:
                    if col and col not in drifted_features:
                        drifted_features.append(col)

        # Dataset drift standard rule: > 50% columns drifted
        drift_detected = drift_share > 0.5 or (column_count > 0 and (drifted_cols_count / column_count) > 0.5)

    except Exception as e:
        logger.warning(f"Could not parse detailed Evidently snapshot metrics: {e}")

    return {
        "dataset_drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "number_of_drifted_columns": drifted_cols_count,
        "drifted_features": drifted_features
    }


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_filename: str = "drift_report.html",
    data_source: str = "cohort"
) -> Dict[str, Any]:
    """
    Compares baseline reference dataset against current production inference logs
    using Evidently AI and generates an interactive HTML report and JSON summary.
    """
    os.makedirs(settings.REPORTS_PATH, exist_ok=True)
    report_path = os.path.join(settings.REPORTS_PATH, report_filename)
    summary_path = os.path.join(settings.REPORTS_PATH, "drift_summary.json")

    logger.info("--> Running Evidently AI Data Drift & Data Summary suite...")

    # Align columns between reference and current datasets
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    ref_aligned = reference_df[common_cols].copy()
    curr_aligned = current_df[common_cols].copy()

    # Initialize Evidently Report with presets
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    # Run comparison
    snapshot = report.run(reference_data=ref_aligned, current_data=curr_aligned)

    # Save HTML interactive report
    snapshot.save_html(report_path)
    logger.info(f"--> Saved interactive Evidently report to: {report_path}")

    # Extract structured KPIs
    metrics_info = extract_drift_metrics_from_snapshot(snapshot, len(common_cols))

    summary = {
        "report_file": report_filename,
        "report_generated": True,
        "last_generated": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(ref_aligned),
        "current_rows": len(curr_aligned),
        "total_features_monitored": len(common_cols),
        "dataset_drift_detected": metrics_info["dataset_drift_detected"],
        "drift_share": metrics_info["drift_share"],
        "number_of_drifted_columns": metrics_info["number_of_drifted_columns"],
        "drifted_features": metrics_info["drifted_features"],
        "data_source": data_source
    }

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist drift summary JSON: {e}")

    return summary


def get_latest_drift_summary() -> Optional[Dict[str, Any]]:
    """Loads the cached drift summary JSON if present."""
    summary_path = os.path.join(settings.REPORTS_PATH, "drift_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def __getattr__(name: str):
    if name == "SessionLocal":
        return db_mod.SessionLocal
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

