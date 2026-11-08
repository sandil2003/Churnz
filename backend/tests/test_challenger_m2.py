"""
Milestone M2 Challenger Test Suite: Serving API & Observability Hardening.
Empirically stress-tests:
1. Malformed JSON payloads, missing fields, out-of-bound types & literals against /predict.
2. Extreme numerical inputs (negative charges, astronomical values, zero tenure).
3. Adversarial CSV batch uploads (empty files, header-only, corrupted values, missing columns).
4. Observability headers (X-Request-ID propagation/UUIDv4 generation, X-Process-Time-Ms precision).
5. Concurrency stress bursts under simultaneous load.
6. Boundary conditions on query parameters (/history pagination, /reports/drift regex).
"""

import io
import re
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "customer_id": "ADV-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
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
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0
}


# =========================================================================
# 1. ADVERSARIAL INPUTS AGAINST /predict
# =========================================================================

def test_predict_empty_and_malformed_json_body():
    """Verify empty bodies, non-JSON strings, and empty objects produce 422."""
    res1 = client.post("/predict", content=b"", headers={"Content-Type": "application/json"})
    assert res1.status_code == 422

    res2 = client.post("/predict", content=b"{{malformed:json", headers={"Content-Type": "application/json"})
    assert res2.status_code == 422

    res3 = client.post("/predict", json={})
    assert res3.status_code == 422


@pytest.mark.parametrize("missing_field", [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges"
])
def test_predict_missing_required_fields(missing_field):
    """Verify omitting any required feature triggers 422 Unprocessable Entity."""
    payload = VALID_PAYLOAD.copy()
    payload.pop(missing_field)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data


@pytest.mark.parametrize("invalid_field,invalid_val", [
    ("gender", "NonBinary"),
    ("SeniorCitizen", 2),
    ("SeniorCitizen", -1),
    ("SeniorCitizen", "Yes"),
    ("tenure", -5),
    ("tenure", 101),
    ("tenure", "twenty"),
    ("Contract", "Six month"),
    ("PaymentMethod", "Cryptocurrency"),
    ("InternetService", "Satellite"),
    ("MonthlyCharges", -10.0),
    ("MonthlyCharges", "Free"),
    ("PaperlessBilling", "Maybe"),
])
def test_predict_schema_type_and_literal_violations(invalid_field, invalid_val):
    """Verify out-of-bound literals and invalid types are rejected with 422."""
    payload = VALID_PAYLOAD.copy()
    payload[invalid_field] = invalid_val
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_extreme_numerical_charges():
    """
    Stress-test numerical extremes:
    - MonthlyCharges = 0.0 (edge of domain)
    - TotalCharges = 100,000,000.0 (ultra-whale customer)
    - TotalCharges = 0.0
    - TotalCharges = ' ' (whitespace string per Kaggle dataset)
    """
    payload_zero_monthly = VALID_PAYLOAD.copy()
    payload_zero_monthly["MonthlyCharges"] = 0.0
    payload_zero_monthly["TotalCharges"] = 0.0
    res0 = client.post("/predict", json=payload_zero_monthly)
    assert res0.status_code == 200
    assert 0.0 <= res0.json()["churn_probability"] <= 1.0

    payload_whale = VALID_PAYLOAD.copy()
    payload_whale["TotalCharges"] = 100_000_000.0
    res_whale = client.post("/predict", json=payload_whale)
    assert res_whale.status_code == 200
    res_data = res_whale.json()
    assert 0.0 <= res_data["churn_probability"] <= 1.0
    assert res_data["risk_level"] in ["Low", "Medium", "High"]
    for rf in res_data["risk_factors"]:
        assert not (rf["attribution"] != rf["attribution"])  # NaN check

    payload_ws = VALID_PAYLOAD.copy()
    payload_ws["TotalCharges"] = "   "
    res_ws = client.post("/predict", json=payload_ws)
    assert res_ws.status_code == 200
    assert 0.0 <= res_ws.json()["churn_probability"] <= 1.0


def test_predict_extreme_tenure_boundary():
    """Test tenure = 0 and tenure = 100 (domain extrema)."""
    payload_new = VALID_PAYLOAD.copy()
    payload_new["tenure"] = 0
    payload_new["TotalCharges"] = 0.0
    res_new = client.post("/predict", json=payload_new)
    assert res_new.status_code == 200
    assert 0.0 <= res_new.json()["churn_probability"] <= 1.0

    payload_old = VALID_PAYLOAD.copy()
    payload_old["tenure"] = 100
    res_old = client.post("/predict", json=payload_old)
    assert res_old.status_code == 200
    assert 0.0 <= res_old.json()["churn_probability"] <= 1.0


def test_predict_adversarial_customer_id_strings():
    """Ensure SQL injection strings, XSS scripts, and Unicode in customer_id do not crash API."""
    adversarial_ids = [
        "'; DROP TABLE prediction_logs; --",
        "<script>alert('xss')</script>",
        "🚀💎🔥_UNICODE_ID",
        "A" * 500
    ]
    for adv_id in adversarial_ids:
        payload = VALID_PAYLOAD.copy()
        payload["customer_id"] = adv_id
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert res.json()["customer_id"] == adv_id


# =========================================================================
# 2. ADVERSARIAL INPUTS AGAINST /batch-predict (JSON & CSV)
# =========================================================================

def test_batch_predict_json_empty_and_malformed():
    """Verify batch JSON rejection for empty payloads and invalid structure."""
    res1 = client.post("/batch-predict", json=[])
    assert res1.status_code == 422

    res2 = client.post("/batch-predict", json={"customers": []})
    assert res2.status_code == 422

    res3 = client.post("/batch-predict", json={"customers": "not-a-list"})
    assert res3.status_code == 422

    res4 = client.post("/batch-predict", json=42)
    assert res4.status_code == 422


def test_batch_predict_json_atomic_validation():
    """If 1 customer out of 5 in a batch is invalid, the entire batch must fail with 422."""
    customers = [VALID_PAYLOAD.copy() for _ in range(5)]
    customers[3]["MonthlyCharges"] = -99.0  # Invalid row
    res = client.post("/batch-predict", json={"customers": customers})
    assert res.status_code == 422
    assert "Row 3" in res.json()["detail"]


def test_batch_predict_csv_empty_file():
    """Verify uploading an empty CSV file returns 422."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res = client.post("/batch-predict", files=files)
    assert res.status_code == 422


def test_batch_predict_csv_header_only():
    """Verify uploading a CSV with only headers (0 rows) returns 422."""
    header_only = "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
    files = {"file": ("headers.csv", io.BytesIO(header_only.encode()), "text/csv")}
    res = client.post("/batch-predict", files=files)
    assert res.status_code == 422


def test_batch_predict_csv_missing_required_columns():
    """
    Vulnerability probe: Uploading CSV missing essential columns like MonthlyCharges and Contract.
    Evaluates whether POST /batch-predict validates CSV against CustomerInput schema.
    """
    broken_csv = (
        "customer_id,gender,SeniorCitizen,tenure\n"
        "CUST-001,Female,0,12\n"
    )
    files = {"file": ("broken.csv", io.BytesIO(broken_csv.encode()), "text/csv")}
    res = client.post("/batch-predict", files=files)
    assert res.status_code == 422, (
        f"Validation bypass bug: CSV missing required columns returned {res.status_code} instead of 422"
    )


def test_batch_predict_csv_corrupted_numerical_values():
    """
    Vulnerability probe: CSV with non-numeric text in numeric fields.
    Evaluates whether POST /batch-predict validates CSV against CustomerInput schema.
    """
    corrupted_csv = (
        "customer_id,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
        "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
        "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "CUST-001,Female,0,No,No,CORRUPTED_TENURE,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.0,140.0\n"
    )
    files = {"file": ("corrupted.csv", io.BytesIO(corrupted_csv.encode()), "text/csv")}
    res = client.post("/batch-predict", files=files)
    assert res.status_code == 422, (
        f"Validation bypass bug: CSV with non-numeric tenure returned {res.status_code} instead of 422"
    )


def test_batch_predict_dedicated_csv_endpoint_validates():
    """Verify that POST /batch-predict/csv strictly validates schema and returns 422 on invalid rows."""
    broken_csv = (
        "customer_id,gender,SeniorCitizen,tenure\n"
        "CUST-001,Female,0,12\n"
    )
    files = {"file": ("broken.csv", io.BytesIO(broken_csv.encode()), "text/csv")}
    res = client.post("/batch-predict/csv", files=files)
    assert res.status_code == 422
    assert "Row 0 invalid" in res.json()["detail"]


def test_batch_predict_csv_auto_generates_missing_customer_id():
    """CSV missing customer_id column should succeed with auto-generated identifiers."""
    valid_csv = (
        "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
        "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
        "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,150.0\n"
        "Male,0,Yes,Yes,48,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Credit card (automatic),55.0,2640.0\n"
    )
    files = {"file": ("no_id.csv", io.BytesIO(valid_csv.encode()), "text/csv")}
    res = client.post("/batch-predict", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert data["predictions"][0]["customer_id"].startswith("BATCH-")
    assert data["predictions"][1]["customer_id"].startswith("BATCH-")


def test_batch_predict_large_csv_stress():
    """Simulate batch inference on 100 customer records in single CSV."""
    header = (
        "customer_id,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
        "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
        "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
    )
    rows = []
    for i in range(100):
        rows.append(
            f"BATCH-{i:04d},Female,0,No,No,{(i % 60) + 1},Yes,No,DSL,No,No,No,No,No,No,"
            f"Month-to-month,Yes,Electronic check,{(i % 70) + 20}.0,{(i % 70 + 20) * ((i % 60) + 1)}.0\n"
        )
    csv_body = header + "".join(rows)

    start_time = time.perf_counter()
    files = {"file": ("large.csv", io.BytesIO(csv_body.encode()), "text/csv")}
    res = client.post("/batch-predict", files=files)
    wall_latency = (time.perf_counter() - start_time) * 1000

    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 100
    assert len(data["predictions"]) == 100
    assert 0.0 <= data["churn_rate"] <= 1.0
    assert 0.0 <= data["avg_probability"] <= 1.0
    assert data["batch_latency_ms"] > 0
    assert data["batch_latency_ms"] <= wall_latency + 50.0


# =========================================================================
# 3. OBSERVABILITY HEADERS (X-Request-ID & X-Process-Time-Ms)
# =========================================================================

UUID_V4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE
)


def test_observability_x_request_id_propagation():
    """Client-provided X-Request-ID must be echoed verbatim in response headers."""
    client_trace_ids = [
        "custom-trace-001",
        "corr-id-alpha-beta-99",
        "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    ]
    for trace_id in client_trace_ids:
        res = client.get("/health", headers={"X-Request-ID": trace_id})
        assert res.status_code == 200
        assert res.headers.get("X-Request-ID") == trace_id


def test_observability_x_request_id_uuid4_generation():
    """When X-Request-ID is omitted by client, middleware must generate a valid UUID v4."""
    seen_ids = set()
    for _ in range(10):
        res = client.get("/health")
        assert res.status_code == 200
        req_id = res.headers.get("X-Request-ID")
        assert req_id is not None
        assert UUID_V4_REGEX.match(req_id) is not None, f"Not a valid UUID v4: {req_id}"
        parsed = uuid.UUID(req_id)
        assert parsed.version == 4
        seen_ids.add(req_id)

    assert len(seen_ids) == 10


def test_observability_x_process_time_ms_accuracy():
    """X-Process-Time-Ms must be positive float and less than or equal to client wall time."""
    start_time = time.perf_counter()
    res = client.post("/predict", json=VALID_PAYLOAD)
    client_duration_ms = (time.perf_counter() - start_time) * 1000

    assert res.status_code == 200
    assert "X-Process-Time-Ms" in res.headers
    process_time = float(res.headers["X-Process-Time-Ms"])
    assert process_time > 0.0
    assert process_time <= client_duration_ms + 10.0


def test_observability_headers_present_on_error_responses():
    """X-Request-ID and X-Process-Time-Ms must be present even on 422 Unprocessable Entity."""
    custom_trace = "err-trace-422"
    res = client.post("/predict", json={}, headers={"X-Request-ID": custom_trace})
    assert res.status_code == 422
    assert res.headers.get("X-Request-ID") == custom_trace
    assert "X-Process-Time-Ms" in res.headers
    assert float(res.headers["X-Process-Time-Ms"]) >= 0.0


# =========================================================================
# 4. CONCURRENCY STRESS TEST
# =========================================================================

def test_concurrency_rapid_burst_predictions():
    """
    Simulate a rapid burst of 40 concurrent requests to /predict.
    Verifies thread safety, absence of deadlocks, unique prediction IDs,
    and consistent TreeSHAP explanations.
    """
    num_requests = 40

    def make_request(idx):
        payload = VALID_PAYLOAD.copy()
        payload["customer_id"] = f"CONCURRENT-{idx:04d}"
        payload["tenure"] = (idx % 60) + 1
        payload["MonthlyCharges"] = round(30.0 + (idx * 1.5), 2)
        payload["TotalCharges"] = round(payload["MonthlyCharges"] * payload["tenure"], 2)

        start = time.perf_counter()
        response = client.post(
            "/predict",
            json=payload,
            headers={"X-Request-ID": f"req-{idx:04d}"}
        )
        elapsed = (time.perf_counter() - start) * 1000
        return idx, response.status_code, response.json(), response.headers, elapsed

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for f in as_completed(futures):
            results.append(f.result())

    assert len(results) == num_requests
    prediction_ids = set()
    for idx, status, body, headers, elapsed in results:
        assert status == 200, f"Request {idx} failed with {status}: {body}"
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["customer_id"] == f"CONCURRENT-{idx:04d}"
        assert headers.get("X-Request-ID") == f"req-{idx:04d}"
        assert float(headers.get("X-Process-Time-Ms", 0)) > 0

        pred_id = body.get("prediction_id")
        assert pred_id is not None
        assert pred_id not in prediction_ids
        prediction_ids.add(pred_id)
        assert len(body["risk_factors"]) > 0


def test_concurrency_mixed_valid_and_invalid_burst():
    """Simulate concurrent storm interleaving valid and invalid requests."""
    def worker(idx):
        if idx % 2 == 0:
            res = client.post("/predict", json=VALID_PAYLOAD)
            return idx, res.status_code
        else:
            bad = VALID_PAYLOAD.copy()
            bad["tenure"] = -1
            res = client.post("/predict", json=bad)
            return idx, res.status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(30)]
        results = [f.result() for f in as_completed(futures)]

    for idx, status in results:
        if idx % 2 == 0:
            assert status == 200
        else:
            assert status == 422


# =========================================================================
# 5. QUERY PARAMETER ADVERSARIAL VALIDATION
# =========================================================================

def test_history_endpoint_query_bounds():
    """Verify limit and offset bounds on /history."""
    res1 = client.get("/history?limit=0")
    assert res1.status_code == 422

    res2 = client.get("/history?limit=501")
    assert res2.status_code == 422

    res3 = client.get("/history?offset=-1")
    assert res3.status_code == 422


def test_reports_drift_format_regex_validation():
    """Verify format parameter rejection for unsupported formats."""
    res_bad = client.get("/reports/drift?format=yaml")
    assert res_bad.status_code == 422
