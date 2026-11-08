import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.config import settings
from src.database import init_db_connection
from src.api.deps import get_model, reload_model_artifact
from src.api import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event handler for FastAPI startup and shutdown.
    Pre-initializes database connection and champion model cache.
    """
    logger.info("Initializing Churnz Serving API...")
    init_db_connection()
    get_model()
    yield
    logger.info("Shutting down Churnz Serving API...")


app = FastAPI(
    title="Telco Churn Prediction MLOps API",
    description="Production-grade ML serving API using LightGBM TreeSHAP, Optuna, MLflow, and Evidently AI",
    version="1.0.0",
    lifespan=lifespan
)

# Observability Middleware: Correlation ID and Latency Tracing
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """
    Adds request correlation ID (X-Request-ID) and processing latency (X-Process-Time-Ms)
    headers to every HTTP response.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(f"Unhandled Exception in request {request_id}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://tools.ietf.org/html/rfc7807",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An internal server error occurred.",
                "error": str(exc),
                "request_id": request_id
            },
            headers={
                "X-Request-ID": request_id,
                "X-Process-Time-Ms": str(latency_ms)
            }
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
    return response


# Global Exception Handlers for RFC 7807 compliance
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception(f"Global Exception Handler: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://tools.ietf.org/html/rfc7807",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An internal server error occurred.",
            "error": str(exc),
            "request_id": request_id
        },
        headers={"X-Request-ID": request_id}
    )


# CORS Configuration
cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8000,*"
)
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if "*" not in cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers at root (backward compatibility) and versioned /api/v1
app.include_router(api_router, prefix="")
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, reload=True)
