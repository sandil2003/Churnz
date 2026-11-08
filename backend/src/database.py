import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class CustomerTable(Base):
    """Raw customer data ingested from Kaggle or data source."""
    __tablename__ = "raw_customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), unique=True, index=True, nullable=False)
    gender = Column(String(20))
    SeniorCitizen = Column(Integer)
    Partner = Column(String(10))
    Dependents = Column(String(10))
    tenure = Column(Integer)
    PhoneService = Column(String(10))
    MultipleLines = Column(String(30))
    InternetService = Column(String(30))
    OnlineSecurity = Column(String(30))
    OnlineBackup = Column(String(30))
    DeviceProtection = Column(String(30))
    TechSupport = Column(String(30))
    StreamingTV = Column(String(30))
    StreamingMovies = Column(String(30))
    Contract = Column(String(30))
    PaperlessBilling = Column(String(10))
    PaymentMethod = Column(String(50))
    MonthlyCharges = Column(Float)
    TotalCharges = Column(Float)
    Churn = Column(String(10), nullable=True)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class PredictionLog(Base):
    """Production inference audit logs with complete traceability."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String(50), index=True, nullable=False)
    features_json = Column(Text, nullable=False)
    churn_prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_version = Column(String(50), index=True, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True, nullable=False)


class GroundTruthTable(Base):
    """Ground truth feedback for model monitoring & drift evaluation."""
    __tablename__ = "ground_truth_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), index=True, nullable=False)
    actual_churn = Column(Integer, nullable=False)
    prediction_id = Column(String(36), index=True, nullable=True)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


# Engine & Session Factory
engine = None
SessionLocal = None


def init_db_connection(db_url: Optional[str] = None) -> bool:
    """Initializes SQLAlchemy engine and creates tables if database is available."""
    global engine, SessionLocal
    target_url = db_url or settings.DATABASE_URL
    try:
        if target_url.startswith("sqlite"):
            engine = create_engine(target_url, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(
                target_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=1800
            )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully connected to database and verified tables.")
        return True
    except Exception as e:
        logger.warning(f"Could not connect to database ({target_url}): {e}. Running in standalone mode.")
        return False


def get_db():
    """Dependency helper for FastAPI endpoints with guaranteed cleanup."""
    if SessionLocal is None:
        init_db_connection()
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def log_prediction_to_db(
    customer_dict: dict,
    prediction: int,
    probability: float,
    risk_level: str,
    prediction_id: Optional[str] = None,
    model_version: Optional[str] = None,
    latency_ms: Optional[float] = None
) -> Optional[str]:
    """
    Utility to persist a single prediction event to PostgreSQL.
    Guarantees session closure in finally block to prevent connection pool leaks.

    Returns:
        Optional[str]: Generated or confirmed prediction_id UUID string.
    """
    if SessionLocal is None:
        init_db_connection()
    if SessionLocal is None:
        return None

    db = SessionLocal()
    try:
        pred_id = prediction_id or str(uuid.uuid4())
        cid = customer_dict.get("customer_id", "UNKNOWN")
        clean_features = {k: v for k, v in customer_dict.items() if k != "customer_id"}

        record = PredictionLog(
            prediction_id=pred_id,
            customer_id=cid,
            features_json=json.dumps(clean_features, default=str, sort_keys=True),
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=risk_level,
            model_version=model_version,
            latency_ms=latency_ms
        )
        db.add(record)
        db.commit()
        return pred_id
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log prediction to DB: {e}")
        return None
    finally:
        db.close()


def log_batch_predictions_to_db(
    records: List[Dict[str, Any]],
    model_version: Optional[str] = None
) -> int:
    """
    Persist multiple prediction events in a single atomic database transaction.
    Guarantees session closure in finally block to prevent connection leaks.

    Args:
        records: List of dicts, each containing:
            - features: dict (or input customer features)
            - churn_prediction: int
            - churn_probability: float
            - risk_level: str
            - Optional[customer_id]: str
            - Optional[prediction_id]: str
            - Optional[model_version]: str
            - Optional[latency_ms]: float
        model_version: Fallback model version if not specified in record.

    Returns:
        int: Number of records successfully persisted.
    """
    if not records:
        return 0
    if SessionLocal is None:
        init_db_connection()
    if SessionLocal is None:
        return 0

    db = SessionLocal()
    try:
        objects = []
        for r in records:
            features = r.get("features") or r.get("customer_dict") or {}
            cid = r.get("customer_id") or (features.get("customer_id") if isinstance(features, dict) else None) or "UNKNOWN"
            pred_id = r.get("prediction_id") or str(uuid.uuid4())
            version = r.get("model_version") or model_version
            latency = r.get("latency_ms")

            clean_features = (
                {k: v for k, v in features.items() if k != "customer_id"}
                if isinstance(features, dict)
                else features
            )

            obj = PredictionLog(
                prediction_id=pred_id,
                customer_id=cid,
                features_json=json.dumps(clean_features, default=str, sort_keys=True),
                churn_prediction=int(r["churn_prediction"]),
                churn_probability=float(r["churn_probability"]),
                risk_level=r["risk_level"],
                model_version=version,
                latency_ms=latency
            )
            objects.append(obj)

        db.add_all(objects)
        db.commit()
        logger.info(f"Successfully logged batch of {len(objects)} predictions to database.")
        return len(objects)
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log batch predictions to DB: {e}")
        return 0
    finally:
        db.close()
