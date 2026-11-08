import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.config import settings
from src.schemas import ModelStatsResponse, PredictionHistoryItem
from src.api.deps import get_model, get_db
from src.database import PredictionLog
from src.monitoring import (
    generate_drift_report,
    load_reference_data,
    load_production_cohort,
    get_latest_drift_summary
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring & Analytics"])


@router.get("/stats", summary="Model Performance Metrics & Feature Importance")
def get_model_stats():
    """Returns champion model training metrics, top global feature importances, and metadata."""
    artifact = get_model()
    if artifact is None:
        return {"status": "no_model_loaded", "message": "Champion model artifact not loaded."}

    metrics = artifact.get("metrics", {})
    top_features = artifact.get("top_features", [])
    feature_columns = artifact.get("feature_columns", [])
    model_version = artifact.get("model_version", "champion-v1.0.0")

    return {
        "model_name": "LightGBM Classifier",
        "model_version": model_version,
        "metrics": metrics,
        "top_features": top_features,
        "features_count": len(feature_columns)
    }


@router.get("/history", response_model=List[PredictionHistoryItem], summary="Inference Prediction History")
def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Records offset for pagination"),
    db=Depends(get_db)
):
    """Fetches recent prediction logs from PostgreSQL with pagination support."""
    if db is None:
        logger.info("Database offline: returning empty inference history.")
        return []

    try:
        records = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            PredictionHistoryItem(
                id=r.id,
                prediction_id=r.prediction_id,
                customer_id=r.customer_id,
                churn_prediction=r.churn_prediction,
                churn_probability=r.churn_probability,
                risk_level=r.risk_level,
                model_version=r.model_version,
                latency_ms=r.latency_ms,
                created_at=r.created_at.isoformat() if r.created_at else None
            )
            for r in records
        ]
    except Exception as e:
        logger.error(f"Error fetching prediction history from DB: {e}")
        return []


@router.get("/reports/drift", summary="Evidently AI Data Drift Report")
def view_drift_report(
    format: str = Query(default="html", pattern="^(html|json)$", description="Response format: html or json"),
    db=Depends(get_db)
):
    """
    Serves the latest Evidently AI data drift report as interactive HTML
    or structured JSON summary. If report does not exist on disk, generates one on-demand.
    """
    report_file = os.path.join(settings.REPORTS_PATH, "drift_report.html")

    if not os.path.exists(report_file):
        logger.info("Drift report not found on disk. Generating on-demand...")
        ref_df = load_reference_data()
        artifact = get_model()
        pipeline = artifact.get("pipeline") if artifact else None

        if ref_df is not None:
            current_df, source = load_production_cohort(min_rows=100, db_session=db, pipeline=pipeline)
            generate_drift_report(
                reference_df=ref_df,
                current_df=current_df,
                report_filename="drift_report.html",
                data_source=source
            )
        else:
            if format == "json":
                return JSONResponse(
                    status_code=404,
                    content={"error": "Reference data not found. Train champion model first."}
                )
            return HTMLResponse(
                content="<h3>Reference data not found. Please train model first via scripts/run_training.py.</h3>",
                status_code=404
            )

    if format == "json":
        summary = get_latest_drift_summary()
        if summary:
            return summary
        return {
            "report_file": "drift_report.html",
            "report_exists": True,
            "status": "Report generated but JSON summary was not parsed."
        }

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to read drift report HTML: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load drift report: {e}")


@router.get("/monitoring/summary", summary="Evidently Drift KPI Summary (JSON)")
def get_monitoring_summary(db=Depends(get_db)):
    """Provides structured drift KPIs for frontend widgets and executive dashboards."""
    summary = get_latest_drift_summary()
    if summary is not None:
        return summary

    # If no cached summary, run on-demand evaluation
    ref_df = load_reference_data()
    artifact = get_model()
    pipeline = artifact.get("pipeline") if artifact else None

    if ref_df is not None:
        current_df, source = load_production_cohort(min_rows=100, db_session=db, pipeline=pipeline)
        summary = generate_drift_report(
            reference_df=ref_df,
            current_df=current_df,
            report_filename="drift_report.html",
            data_source=source
        )
        return summary

    return {
        "report_exists": False,
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "total_features_monitored": 0,
        "drifted_features": []
    }
