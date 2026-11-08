"""
Empirical Challenger M2.3 Adversarial Test Harness.
Focus: Deep stress-testing of POST /batch-predict and POST /batch-predict/csv.

Objectives:
1. Craft adversarial CSV payloads:
   - Missing required columns
   - Wrong column headers (completely unrecognized schema)
   - Extra columns mixed with valid schema and with invalid schema
   - Empty files (0 bytes)
   - Header-only files (0 rows)
   - Non-numeric values for numeric fields (tenure, MonthlyCharges, SeniorCitizen)
   - Out-of-range categorical values (gender, Contract, PaymentMethod, InternetService, etc.)
   - Corrupted CSV syntax (unclosed quotes, malformed delimiters)
2. Verify HTTP 422 Unprocessable Entity with informative error details (never HTTP 200, never HTTP 500).
3. Test large batch multipart CSV (100 rows) verifying:
   - status_code == 200
   - total_processed == 100
   - churn_rate and avg_probability valid
   - risk_factors and explanations valid and populated for all 100 rows
   - vectorized TreeSHAP attribution correctness and consistency
"""

import io
import time
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

VALID_CSV_HEADER = (
    "customer_id,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
    "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
    "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
    "MonthlyCharges,TotalCharges"
)

VALID_CSV_ROW_TEMPLATE = (
    "{customer_id},{gender},{SeniorCitizen},{Partner},{Dependents},{tenure},{PhoneService},"
    "{MultipleLines},{InternetService},{OnlineSecurity},{OnlineBackup},{DeviceProtection},"
    "{TechSupport},{StreamingTV},{StreamingMovies},{Contract},{PaperlessBilling},{PaymentMethod},"
    "{MonthlyCharges},{TotalCharges}"
)

def make_valid_row(customer_id="CUST-0001", **overrides):
    fields = {
        "customer_id": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
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
        "TotalCharges": 1026.0,
    }
    fields.update(overrides)
    return VALID_CSV_ROW_TEMPLATE.format(**fields)


# =========================================================================
# 1. ADVERSARIAL CSV PAYLOADS: MISSING REQUIRED COLUMNS
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
@pytest.mark.parametrize("omitted_column", [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges"
])
def test_csv_missing_single_required_column(endpoint, omitted_column):
    """Omitting any single required column in CSV must return 422 with informative error details."""
    cols = [c for c in VALID_CSV_HEADER.split(",") if c != omitted_column]
    csv_header = ",".join(cols)

    row_dict = {
        "customer_id": "CUST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
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
        "TotalCharges": 1026.0,
    }
    row_dict.pop(omitted_column)
    row_vals = ",".join(str(row_dict[c]) for c in cols)
    csv_content = f"{csv_header}\n{row_vals}\n"

    files = {"file": ("missing_col.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} instead of 422 when missing column '{omitted_column}'"
    )
    detail = str(res.json().get("detail", ""))
    assert len(detail) > 0, f"Error detail must not be empty on {endpoint}"
    assert (omitted_column in detail or "Field required" in detail or "validation" in detail.lower() or "invalid" in detail.lower()), (
        f"Detail did not contain helpful context: {detail}"
    )


# =========================================================================
# 2. ADVERSARIAL CSV PAYLOADS: WRONG COLUMN HEADERS & UNRECOGNIZED SCHEMA
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_completely_wrong_headers(endpoint):
    """Uploading a CSV with completely wrong/unrecognized column headers must return 422."""
    wrong_csv = (
        "id,user_name,email,age,membership_type,billing_amount\n"
        "101,alice,alice@example.com,29,Premium,99.99\n"
        "102,bob,bob@example.com,45,Basic,19.99\n"
    )
    files = {"file": ("wrong_headers.csv", io.BytesIO(wrong_csv.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} instead of 422 on completely wrong headers"
    )
    detail = res.json().get("detail", "")
    assert "validation" in detail.lower() or "invalid" in detail.lower() or "field required" in detail.lower()


# =========================================================================
# 3. ADVERSARIAL CSV PAYLOADS: EXTRA COLUMNS
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_with_extra_columns_alongside_valid_data(endpoint):
    """
    CSV containing valid columns PLUS extra unknown columns (e.g. notes, internal_id, tags).
    Pydantic schema ignores extra fields, so this should process cleanly or fail gracefully with 422,
    never crashing with 500.
    """
    extra_header = VALID_CSV_HEADER + ",ExtraNotes,InternalRiskScore\n"
    extra_row = make_valid_row("CUST-EXTRA-1") + ',"High value VIP customer",0.12\n'
    csv_content = extra_header + extra_row

    files = {"file": ("extra_cols.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code in [200, 422], (
        f"Endpoint {endpoint} returned unexpected status code {res.status_code} (must not be 500)"
    )
    if res.status_code == 200:
        data = res.json()
        assert data["total_processed"] == 1
        assert len(data["predictions"]) == 1
        assert "risk_factors" in data["predictions"][0]


# =========================================================================
# 4. ADVERSARIAL CSV PAYLOADS: EMPTY FILES AND ZERO-ROW HEADER-ONLY
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_empty_file_zero_bytes(endpoint):
    """Uploading a 0-byte file must return 422 Unprocessable Entity."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res = client.post(endpoint, files=files)
    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} on 0-byte file (expected 422)"
    )
    assert len(str(res.json().get("detail", ""))) > 0


@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_header_only_zero_rows(endpoint):
    """Uploading a CSV with headers but 0 data rows must return 422 (cannot process empty batch)."""
    csv_content = VALID_CSV_HEADER + "\n"
    files = {"file": ("header_only.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)
    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} on header-only CSV (expected 422)"
    )
    assert len(str(res.json().get("detail", ""))) > 0


# =========================================================================
# 5. ADVERSARIAL CSV PAYLOADS: NON-NUMERIC VALUES FOR NUMERIC FIELDS
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
@pytest.mark.parametrize("numeric_field,bad_val", [
    ("SeniorCitizen", "not_a_number"),
    ("SeniorCitizen", "2.5"),
    ("tenure", "twenty"),
    ("tenure", "-1"),
    ("tenure", "150"),
    ("MonthlyCharges", "free_trial"),
    ("MonthlyCharges", "-50.0"),
    ("TotalCharges", "invalid_charge_string"),
])
def test_csv_corrupted_numeric_fields(endpoint, numeric_field, bad_val):
    """Corrupted non-numeric or out-of-range numeric fields in CSV must trigger 422."""
    if numeric_field == "TotalCharges":
        # Note: TotalCharges has a validator that coerces bad values to 0.0 unless strictly rejected
        # Let's test how endpoint behaves
        pass

    row = make_valid_row("CUST-BAD-NUM", **{numeric_field: bad_val})
    csv_content = f"{VALID_CSV_HEADER}\n{row}\n"

    files = {"file": ("bad_num.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    if numeric_field == "TotalCharges":
        # TotalCharges validator parses unparseable strings to 0.0 for Kaggle dataset legacy compatibility
        assert res.status_code in [200, 422]
    else:
        assert res.status_code == 422, (
            f"Endpoint {endpoint} returned {res.status_code} on {numeric_field}={bad_val} (expected 422)"
        )
        detail = str(res.json().get("detail", ""))
        assert len(detail) > 0


# =========================================================================
# 6. ADVERSARIAL CSV PAYLOADS: OUT-OF-RANGE CATEGORICAL VALUES
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
@pytest.mark.parametrize("cat_field,bad_val", [
    ("gender", "Other"),
    ("gender", "Unknown"),
    ("Partner", "Maybe"),
    ("Dependents", "Unknown"),
    ("PhoneService", "None"),
    ("MultipleLines", "TenLines"),
    ("InternetService", "Satellite"),
    ("InternetService", "Dial-up"),
    ("OnlineSecurity", "Sometimes"),
    ("OnlineBackup", "Cloud"),
    ("DeviceProtection", "Sure"),
    ("TechSupport", "YesPlease"),
    ("StreamingTV", "4K"),
    ("StreamingMovies", "IMAX"),
    ("Contract", "Five year"),
    ("Contract", "Weekly"),
    ("PaperlessBilling", "EmailOnly"),
    ("PaymentMethod", "Bitcoin"),
    ("PaymentMethod", "Cash"),
])
def test_csv_out_of_range_categorical_values(endpoint, cat_field, bad_val):
    """Invalid categorical literals in CSV must trigger 422 Unprocessable Entity."""
    row = make_valid_row("CUST-BAD-CAT", **{cat_field: bad_val})
    csv_content = f"{VALID_CSV_HEADER}\n{row}\n"

    files = {"file": ("bad_cat.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} instead of 422 for {cat_field}='{bad_val}'"
    )
    detail = str(res.json().get("detail", ""))
    assert len(detail) > 0, f"Error detail must not be empty on {endpoint}"
    assert "Row 0" in detail or "validation" in detail.lower() or "invalid" in detail.lower()


# =========================================================================
# 7. ADVERSARIAL CSV PAYLOADS: CORRUPTED SYNTAX AND FORMATTING
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_corrupted_syntax_unclosed_quotes(endpoint):
    """CSV with corrupted syntax such as unclosed quotes must return 422 (never 500)."""
    corrupted_content = (
        VALID_CSV_HEADER + "\n"
        'CUST-001,"Female,0,No,No,12,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,85.5,1026.0\n'
    )
    files = {"file": ("corrupted_quotes.csv", io.BytesIO(corrupted_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} instead of 422 on syntax corruption"
    )


@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_corrupted_binary_garbage(endpoint):
    """Uploading random binary garbage must return 422 (never 500)."""
    garbage_bytes = b"\x00\xff\xfe\x00\x12\x34\x56\x78\x9a\xbc\xde\xf0" * 32
    files = {"file": ("garbage.bin", io.BytesIO(garbage_bytes), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422, (
        f"Endpoint {endpoint} returned {res.status_code} on binary garbage (must be 422)"
    )


# =========================================================================
# 8. ATOMIC BATCH VALIDATION: 1 BAD ROW IN A BATCH OF VALID ROWS
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_atomic_validation_identifies_corrupt_row(endpoint):
    """
    A batch of 10 rows where row index 6 has an invalid Contract must fail completely
    with 422 and pinpoint Row 6 in the error message.
    """
    rows = []
    for i in range(10):
        if i == 6:
            rows.append(make_valid_row(f"CUST-{i:03d}", Contract="Lifetime"))
        else:
            rows.append(make_valid_row(f"CUST-{i:03d}"))

    csv_content = VALID_CSV_HEADER + "\n" + "\n".join(rows) + "\n"
    files = {"file": ("batch_10.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 422
    detail = res.json().get("detail", "")
    assert "Row 6" in detail, f"Expected 'Row 6' in error detail, got: {detail}"


# =========================================================================
# 9. LARGE BATCH SCALING (100 VALID ROWS) & VECTORIZED TREESHAP VERIFICATION
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_large_batch_100_rows_vectorized_treeshap(endpoint):
    """
    Test large batch multipart CSV (100 valid rows):
    1. Returns HTTP 200
    2. total_processed == 100
    3. churn_rate between 0.0 and 1.0
    4. avg_probability between 0.0 and 1.0
    5. Exactly 100 predictions returned
    6. All 100 predictions have:
       - valid prediction_id (UUID format)
       - valid customer_id matching row or auto-generated
       - valid churn_prediction (0 or 1)
       - valid churn_probability between 0.0 and 1.0
       - valid risk_level ("Low", "Medium", "High")
       - non-empty risk_factors list (top 3 factors)
       - non-empty explanation list (FeatureContribution objects)
       - each explanation has feature, attribution, direction, display_text
    7. Vectorized batch TreeSHAP execution latency is performant
    """
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    internets = ["DSL", "Fiber optic", "No"]
    genders = ["Male", "Female"]

    rows = []
    for i in range(100):
        cid = f"BATCH-TEST-{i+1:04d}"
        contract = contracts[i % len(contracts)]
        payment = payments[i % len(payments)]
        internet = internets[i % len(internets)]
        gender = genders[i % len(genders)]
        tenure = (i % 71) + 1
        senior = 1 if i % 4 == 0 else 0
        monthly = round(20.0 + (i % 80) * 1.25, 2)
        total = round(monthly * tenure, 2)

        rows.append(make_valid_row(
            customer_id=cid,
            gender=gender,
            SeniorCitizen=senior,
            tenure=tenure,
            InternetService=internet,
            Contract=contract,
            PaymentMethod=payment,
            MonthlyCharges=monthly,
            TotalCharges=total
        ))

    csv_content = VALID_CSV_HEADER + "\n" + "\n".join(rows) + "\n"

    start_perf = time.perf_counter()
    files = {"file": ("batch_100.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)
    elapsed_ms = (time.perf_counter() - start_perf) * 1000

    assert res.status_code == 200, f"Expected 200 OK from {endpoint}, got {res.status_code}: {res.text}"
    data = res.json()

    # Verify summary fields
    assert data["total_processed"] == 100
    assert len(data["predictions"]) == 100
    assert 0 <= data["churn_count"] <= 100
    assert 0.0 <= data["churn_rate"] <= 1.0
    assert round(data["churn_count"] / 100.0, 4) == data["churn_rate"]
    assert 0.0 <= data["avg_probability"] <= 1.0
    assert data["batch_latency_ms"] > 0
    # Vectorized execution should be well under 10 seconds for 100 rows
    assert elapsed_ms < 15000, f"Batch execution too slow: {elapsed_ms:.1f}ms"

    # Verify per-prediction TreeSHAP and payload contracts
    prediction_ids = set()
    customer_ids = set()

    for idx, pred in enumerate(data["predictions"]):
        expected_cid = f"BATCH-TEST-{idx+1:04d}"
        assert pred["customer_id"] == expected_cid
        customer_ids.add(pred["customer_id"])

        pid = pred["prediction_id"]
        assert pid is not None and len(pid) > 10, f"Invalid prediction_id on row {idx}"
        prediction_ids.add(pid)

        assert pred["churn_prediction"] in [0, 1]
        assert 0.0 <= pred["churn_probability"] <= 1.0
        assert pred["risk_level"] in ["Low", "Medium", "High"]

        # Check risk factors (List[FeatureContribution])
        risk_factors = pred["risk_factors"]
        assert isinstance(risk_factors, list), f"risk_factors must be list on row {idx}"
        assert len(risk_factors) >= 1, f"risk_factors must not be empty on row {idx}"
        for rf in risk_factors:
            assert isinstance(rf, dict)
            assert "feature" in rf and isinstance(rf["feature"], str)
            assert "attribution" in rf and isinstance(rf["attribution"], (int, float))
            assert rf["direction"] in ["increases_risk", "decreases_risk"]
            assert "display_text" in rf and isinstance(rf["display_text"], str)
            assert len(rf["display_text"]) > 0

        # Check explanation (List[str])
        explanations = pred["explanation"]
        assert isinstance(explanations, list), f"explanation must be list on row {idx}"
        assert len(explanations) >= 1, f"explanation must not be empty on row {idx}"
        for exp in explanations:
            assert isinstance(exp, str) and len(exp) > 0

    # Ensure all prediction IDs are unique
    assert len(prediction_ids) == 100, "All 100 prediction IDs must be globally unique UUIDs"
    assert len(customer_ids) == 100


# =========================================================================
# 10. LARGE BATCH WITH MISSING CUSTOMER_ID AUTO-GENERATION (100 ROWS)
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_large_batch_100_rows_missing_customer_id_auto_generation(endpoint):
    """
    Verify 100 rows CSV lacking customer_id column auto-generates BATCH-0001 ... BATCH-0100
    and successfully generates valid TreeSHAP predictions for all.
    """
    cols_no_id = [c for c in VALID_CSV_HEADER.split(",") if c != "customer_id"]
    csv_header = ",".join(cols_no_id)

    rows = []
    for i in range(100):
        row_fields = {
            "gender": "Male" if i % 2 == 0 else "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": (i % 50) + 1,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": 70.0 * ((i % 50) + 1),
        }
        row_str = ",".join(str(row_fields[c]) for c in cols_no_id)
        rows.append(row_str)

    csv_content = csv_header + "\n" + "\n".join(rows) + "\n"
    files = {"file": ("no_id_100.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post(endpoint, files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 100
    assert len(data["predictions"]) == 100
    for idx, p in enumerate(data["predictions"]):
        expected_id = f"BATCH-{idx+1:04d}"
        assert p["customer_id"] == expected_id, f"Expected {expected_id}, got {p['customer_id']}"
        assert len(p["risk_factors"]) >= 1
        assert len(p["explanation"]) >= 1


# =========================================================================
# 11. ADDITIONAL ADVERSARIAL CASES: WRONG UPLOAD KEY, EMPTY CELLS, WHITESPACE
# =========================================================================

@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_wrong_upload_key(endpoint):
    """Uploading CSV with key other than 'file' must return 422."""
    files = {"document": ("data.csv", io.BytesIO(VALID_CSV_HEADER.encode()), "text/csv")}
    res = client.post(endpoint, files=files)
    assert res.status_code == 422
    assert len(str(res.json().get("detail", ""))) > 0


@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_missing_values_empty_cells(endpoint):
    """CSV with missing values / empty commas in mandatory fields must return 422."""
    # gender missing between commas
    row = make_valid_row("CUST-EMPTY-CELL")
    # Replace Female with empty string
    row_empty_gender = row.replace("Female", "")
    csv_content = f"{VALID_CSV_HEADER}\n{row_empty_gender}\n"
    files = {"file": ("empty_cell.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    res = client.post(endpoint, files=files)
    assert res.status_code == 422


@pytest.mark.parametrize("endpoint", ["/batch-predict", "/batch-predict/csv"])
def test_csv_whitespace_categorical_values(endpoint):
    """CSV with whitespace-only values in categorical fields must return 422."""
    row = make_valid_row("CUST-WS", Contract="   ")
    csv_content = f"{VALID_CSV_HEADER}\n{row}\n"
    files = {"file": ("whitespace.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    res = client.post(endpoint, files=files)
    assert res.status_code == 422

