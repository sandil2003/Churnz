import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.monitoring import (
    load_reference_data,
    load_production_cohort,
    generate_drift_report,
    get_latest_drift_summary
)

client = TestClient(app)


def test_load_reference_data():
    """Confirms reference training dataset loads with clean feature columns."""
    ref_df = load_reference_data()
    assert ref_df is not None, "Reference data must be present"
    assert len(ref_df) > 0
    assert "target" not in ref_df.columns
    assert "Churn" not in ref_df.columns
    assert "MonthlyCharges" in ref_df.columns
    assert "tenure" in ref_df.columns


def test_load_production_cohort():
    """Verifies production cohort loading with DB query or realistic fallback."""
    ref_df = load_reference_data()
    cohort_df, source = load_production_cohort(min_rows=100)

    assert cohort_df is not None
    assert len(cohort_df) > 0
    assert isinstance(source, str)


def test_generate_drift_report_execution():
    """Verifies that Evidently AI generates both interactive HTML and structured summary JSON."""
    ref_df = load_reference_data()
    assert ref_df is not None

    # Use small slices to ensure fast test execution
    ref_slice = ref_df.iloc[:80].copy()
    curr_slice = ref_df.iloc[80:160].copy()

    test_report_name = "test_pytest_drift_report.html"
    summary = generate_drift_report(
        reference_df=ref_slice,
        current_df=curr_slice,
        report_filename=test_report_name,
        data_source="pytest_test"
    )

    report_path = os.path.join(settings.REPORTS_PATH, test_report_name)
    assert os.path.exists(report_path), f"Report file {report_path} must exist"

    assert summary["report_file"] == test_report_name
    assert summary["reference_rows"] == len(ref_slice)
    assert summary["current_rows"] == len(curr_slice)
    assert "dataset_drift_detected" in summary
    assert isinstance(summary["dataset_drift_detected"], bool)
    assert "drift_share" in summary
    assert 0.0 <= summary["drift_share"] <= 1.0
    assert "number_of_drifted_columns" in summary
    assert "drifted_features" in summary


def test_view_drift_report_html_endpoint():
    """Tests GET /reports/drift serving interactive HTML report."""
    response = client.get("/reports/drift")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert len(response.text) > 100


def test_view_drift_report_json_endpoint():
    """Tests GET /reports/drift?format=json serving structured summary."""
    response = client.get("/reports/drift?format=json")
    assert response.status_code == 200
    data = response.json()
    assert "report_file" in data


def test_monitoring_summary_endpoint():
    """Tests GET /monitoring/summary endpoint for frontend widgets."""
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_drift_detected" in data
    assert "drift_share" in data
    assert "total_features_monitored" in data


def test_drift_detection_sensitivity():
    """Verifies that an intentionally drifted feature is successfully identified."""
    ref_df = load_reference_data()
    assert ref_df is not None

    ref_slice = ref_df.iloc[:100].copy()
    curr_drifted = ref_df.iloc[:100].copy()

    # Dramatically shift MonthlyCharges by $150 to guarantee drift detection
    if "MonthlyCharges" in curr_drifted.columns:
        curr_drifted["MonthlyCharges"] = curr_drifted["MonthlyCharges"] + 150.0

    summary = generate_drift_report(
        reference_df=ref_slice,
        current_df=curr_drifted,
        report_filename="test_drifted_report.html",
        data_source="intentional_drift_test"
    )

    assert summary["number_of_drifted_columns"] >= 1
    assert "MonthlyCharges" in summary["drifted_features"]
