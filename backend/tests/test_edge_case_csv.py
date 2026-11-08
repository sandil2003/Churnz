"""
Edge-Case Multipart CSV Payloads Verification for M2 Challenger.
Verifies that all edge-case multipart CSV payloads return clean HTTP 422/400
and never produce unhandled HTTP 500 errors.
"""

import io
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

VALID_CSV_HEADER = (
    "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
    "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,"
    "StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
    "MonthlyCharges,TotalCharges"
)

EDGE_CASES = [
    ("whitespace_only", "   \n\t  \n   \n"),
    ("header_only_trailing_newlines", VALID_CSV_HEADER + "\n\n\n\n"),
    ("header_only_trailing_spaces", VALID_CSV_HEADER + "   \n   \n"),
    ("header_only_crlf", (VALID_CSV_HEADER + "\r\n\r\n\r\n").encode("utf-8")),
    ("single_comma", ",\n"),
    ("single_column", "customer_id\n"),
    ("commas_only_header", ",,,,,,,,,,,,,,,,,,,,\n"),
    ("header_with_empty_comma_row", VALID_CSV_HEADER + "\n,,,,,,,,,,,,,,,,,,,\n"),
    ("header_with_whitespace_row", VALID_CSV_HEADER + "\n" + ",".join(["   "] * 19) + "\n"),
    ("unclosed_quote", '"gender,Partner\nMale,Yes'),
    ("semicolon_delimited", VALID_CSV_HEADER.replace(",", ";") + "\n"),
    ("zero_bytes", b""),
    ("binary_garbage", b"\x00\xff\xfe\x00\x12\x34\x56\x78\x9a\xbc\xde\xf0"),
    ("huge_header", ("col_" + "x" * 1000 + ",") * 50 + "\n"),
]


@pytest.mark.parametrize("ep", ["/batch-predict", "/batch-predict/csv"])
@pytest.mark.parametrize("name,payload", EDGE_CASES)
def test_edge_case_csv_payloads_return_422(ep, name, payload):
    content = payload.encode("utf-8") if isinstance(payload, str) else payload
    files = {"file": ("edge_test.csv", io.BytesIO(content), "text/csv")}
    res = client.post(ep, files=files)
    assert res.status_code in (400, 422), (
        f"Endpoint {ep} failed on edge-case '{name}': returned {res.status_code} "
        f"with body: {res.text}"
    )
    assert res.status_code != 500, f"Endpoint {ep} returned 500 on edge-case '{name}'"
