import io
import uuid
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

SAMPLE_HIGH_RISK = {
    "customer_id": "TEST-HIGH-01",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
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
    "MonthlyCharges": 95.5,
    "TotalCharges": 95.5
}

SAMPLE_LOW_RISK = {
    "customer_id": "TEST-LOW-01",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "project" in data
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "database_connected" in data
    assert "model_version" in data


def test_versioned_prefix_parity():
    """Confirms both / and /api/v1 routes serve identical functionality."""
    res_root = client.get("/health")
    res_v1 = client.get("/api/v1/health")
    assert res_root.status_code == 200
    assert res_v1.status_code == 200
    assert res_root.json()["model_loaded"] == res_v1.json()["model_loaded"]


def test_predict_endpoint_high_risk_customer():
    response = client.post("/predict", json=SAMPLE_HIGH_RISK)
    assert response.status_code == 200
    res = response.json()
    assert res["customer_id"] == "TEST-HIGH-01"
    assert "prediction_id" in res and res["prediction_id"] is not None
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["churn_prediction"] in (0, 1)
    assert res["risk_level"] in ["Low", "Medium", "High"]
    assert "model_version" in res

    # Authentic TreeSHAP risk factors
    assert "risk_factors" in res
    assert len(res["risk_factors"]) > 0
    first_factor = res["risk_factors"][0]
    assert "feature" in first_factor
    assert "attribution" in first_factor
    assert first_factor["direction"] in ["increases_risk", "decreases_risk"]
    assert "display_text" in first_factor

    # Backward-compatible explanation
    assert "explanation" in res
    assert len(res["explanation"]) > 0


def test_predict_endpoint_low_risk_customer():
    response = client.post("/predict", json=SAMPLE_LOW_RISK)
    assert response.status_code == 200
    res = response.json()
    assert res["customer_id"] == "TEST-LOW-01"
    assert res["churn_probability"] < 0.5
    assert res["churn_prediction"] == 0
    assert res["risk_level"] == "Low"

    # Confirms retention factors identified
    directions = [f["direction"] for f in res["risk_factors"]]
    assert "decreases_risk" in directions


def test_batch_predict_json():
    payload = {"customers": [SAMPLE_HIGH_RISK, SAMPLE_LOW_RISK]}
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["total_processed"] == 2
    assert "churn_count" in res
    assert 0.0 <= res["churn_rate"] <= 1.0
    assert "avg_probability" in res
    assert res["batch_latency_ms"] >= 0.0
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["prediction_id"] is not None


def test_batch_predict_csv_upload():
    # Build CSV in memory
    csv_content = (
        "customer_id,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
        "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
        "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "CSV-001,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,150.0\n"
        "CSV-002,Male,0,Yes,Yes,48,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Credit card (automatic),55.0,2640.0\n"
    )
    files = {"file": ("batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/batch-predict", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["total_processed"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["customer_id"] == "CSV-001"
    assert res["predictions"][1]["customer_id"] == "CSV-002"


def test_feedback_endpoint():
    feedback_payload = {
        "customer_id": "TEST-CUST-FEEDBACK",
        "actual_churn": 1,
        "prediction_id": str(uuid.uuid4())
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] in ["ok", "success"]
    assert res["customer_id"] == "TEST-CUST-FEEDBACK"


def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    res = response.json()
    assert "metrics" in res
    assert "top_features" in res
    assert "features_count" in res
    assert res["features_count"] == 36


def test_history_endpoint():
    response = client.get("/history?limit=10&offset=0")
    assert response.status_code == 200
    res = response.json()
    assert isinstance(res, list)


def test_observability_headers():
    custom_trace_id = "trace-custom-9999"
    response = client.get("/health", headers={"X-Request-ID": custom_trace_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_trace_id
    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0


def test_validation_error_422():
    # Negative MonthlyCharges should fail Pydantic validation
    invalid_customer = SAMPLE_HIGH_RISK.copy()
    invalid_customer["MonthlyCharges"] = -50.0
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422
