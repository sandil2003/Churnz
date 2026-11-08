from fastapi import APIRouter
from src.config import settings
from src.schemas import HealthResponse
from src.api.deps import get_model, is_db_connected

router = APIRouter(tags=["Health"])


@router.get("/", summary="Root Project Information")
def root():
    """Provides platform overview and available API endpoints."""
    return {
        "project": "Telco Customer Churn MLOps Platform",
        "status": "healthy",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "/health",
            "/predict",
            "/batch-predict",
            "/feedback",
            "/stats",
            "/history",
            "/reports/drift"
        ]
    }


@router.get("/health", response_model=HealthResponse, summary="Service & Model Healthcheck")
def healthcheck():
    """Returns live status of ML serving components, model load status, and database."""
    artifact = get_model()
    model_loaded = artifact is not None
    db_connected = is_db_connected()

    model_version = "champion-v1.0.0"
    if artifact and isinstance(artifact, dict):
        model_version = artifact.get("model_version", "champion-v1.0.0")

    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_version=model_version,
        database_connected=db_connected,
        model_path=settings.MODEL_PATH
    )
