import uuid
import json
import pytest
from pydantic import ValidationError
from sqlalchemy import inspect
from src.config import Settings
from src.database import (
    Base,
    PredictionLog,
    CustomerTable,
    GroundTruthTable,
    init_db_connection,
    log_prediction_to_db,
    log_batch_predictions_to_db,
    get_db
)

@pytest.fixture(autouse=True)
def setup_test_db():
    """Initializes in-memory SQLite database for isolated test execution."""
    success = init_db_connection("sqlite:///:memory:")
    assert success is True
    yield
    # Cleanup after test
    import src.database as db_mod
    if db_mod.engine:
        Base.metadata.drop_all(bind=db_mod.engine)
        db_mod.engine.dispose()
    db_mod.engine = None
    db_mod.SessionLocal = None


def test_config_database_url_upgrade():
    """Verifies that postgresql:// and postgres:// are auto-upgraded to postgresql+psycopg2://."""
    s1 = Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/testdb")
    assert s1.DATABASE_URL == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

    s2 = Settings(DATABASE_URL="postgres://user:pass@localhost:5432/testdb")
    assert s2.DATABASE_URL == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

    s3 = Settings(DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/testdb")
    assert s3.DATABASE_URL == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

    s4 = Settings(DATABASE_URL="sqlite:///:memory:")
    assert s4.DATABASE_URL == "sqlite:///:memory:"


def test_database_schema_columns():
    """Verifies all required audit columns exist on PredictionLog."""
    import src.database as db_mod
    inspector = inspect(db_mod.engine)
    columns = {col["name"] for col in inspector.get_columns("prediction_logs")}

    required_columns = {
        "id",
        "prediction_id",
        "customer_id",
        "features_json",
        "churn_prediction",
        "churn_probability",
        "risk_level",
        "model_version",
        "latency_ms",
        "created_at"
    }
    assert required_columns.issubset(columns)


def test_log_single_prediction_with_audit_fields():
    """Verifies single prediction logging with UUID prediction_id, model_version, and latency_ms."""
    import src.database as db_mod
    customer_dict = {
        "customer_id": "TEST-CUST-001",
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.5
    }
    pred_id = log_prediction_to_db(
        customer_dict=customer_dict,
        prediction=1,
        probability=0.78,
        risk_level="High",
        model_version="champion-lgbm-v1",
        latency_ms=14.2
    )

    assert pred_id is not None
    # Validate UUID format
    val = uuid.UUID(pred_id, version=4)
    assert str(val) == pred_id

    db = db_mod.SessionLocal()
    try:
        record = db.query(PredictionLog).filter_by(prediction_id=pred_id).first()
        assert record is not None
        assert record.customer_id == "TEST-CUST-001"
        assert record.churn_prediction == 1
        assert pytest.approx(record.churn_probability, 0.001) == 0.78
        assert record.risk_level == "High"
        assert record.model_version == "champion-lgbm-v1"
        assert pytest.approx(record.latency_ms, 0.01) == 14.2
        # Ensure customer_id was stripped from features_json
        parsed = json.loads(record.features_json)
        assert "customer_id" not in parsed
        assert parsed["Contract"] == "Month-to-month"
    finally:
        db.close()


def test_log_batch_predictions_to_db():
    """Verifies batch logging persists multiple records atomically in a single transaction."""
    import src.database as db_mod
    batch_records = [
        {
            "customer_id": f"BATCH-{i:03d}",
            "features": {"tenure": i * 10, "MonthlyCharges": 50.0 + i},
            "churn_prediction": 1 if i % 2 == 0 else 0,
            "churn_probability": 0.2 * i,
            "risk_level": "High" if i >= 3 else "Low",
            "latency_ms": 2.5
        }
        for i in range(5)
    ]

    count = log_batch_predictions_to_db(batch_records, model_version="batch-v1")
    assert count == 5

    db = db_mod.SessionLocal()
    try:
        total = db.query(PredictionLog).filter(PredictionLog.model_version == "batch-v1").count()
        assert total == 5
    finally:
        db.close()


def test_get_db_dependency_closes_session():
    """Verifies FastAPI get_db generator properly closes session in finally block."""
    gen = get_db()
    db = next(gen)
    assert db is not None
    # Generator terminates via next/StopIteration, which triggers finally: db.close()
    with pytest.raises(StopIteration):
        next(gen)
