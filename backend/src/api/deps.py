import os
import joblib
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException
from src.config import settings
import src.database as db_mod
from src.database import get_db, init_db_connection

logger = logging.getLogger(__name__)

# Global model cache
_cached_artifact: Optional[Dict[str, Any]] = None


def load_model_from_mlflow_registry() -> Optional[Dict[str, Any]]:
    """
    Attempts to load the active champion model from the MLflow Model Registry.
    Loads the LightGBM booster registered under settings.MLFLOW_MODEL_NAME@settings.MLFLOW_MODEL_ALIAS
    and marries it with the fitted TelcoFeaturePipeline and training metrics.
    """
    try:
        import mlflow
        import mlflow.lightgbm
        from src.train import is_remote_mlflow_reachable
        from src.config import BASE_DIR

        tracking_uri = settings.MLFLOW_TRACKING_URI
        if not is_remote_mlflow_reachable(tracking_uri):
            db_file = str(BASE_DIR / "mlflow.db").replace("\\", "/")
            tracking_uri = f"sqlite:///{db_file}"
        mlflow.set_tracking_uri(tracking_uri)

        model_name = getattr(settings, "MLFLOW_MODEL_NAME", "TelcoChurnChampion")
        alias = getattr(settings, "MLFLOW_MODEL_ALIAS", "champion")
        client = mlflow.tracking.MlflowClient()

        # Check if model exists in registry by alias or latest version
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            model_uri = f"models:/{model_name}@{alias}"
            version_str = f"mlflow-{model_name}-v{mv.version}"
        except Exception:
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                return None
            mv = latest_versions[0]
            model_uri = f"models:/{model_name}/{mv.version}"
            version_str = f"mlflow-{model_name}-v{mv.version}"

        lgb_model = mlflow.lightgbm.load_model(model_uri)
        logger.info(f"Loaded champion model directly from MLflow Model Registry ({model_uri})")

        # Load feature pipeline & metrics from disk artifact to form complete inference package
        pipeline = None
        metrics = {}
        top_features = []
        feature_columns = []
        if os.path.exists(settings.MODEL_PATH):
            disk_art = joblib.load(settings.MODEL_PATH)
            pipeline = disk_art.get("pipeline")
            metrics = disk_art.get("metrics", {})
            top_features = disk_art.get("top_features", [])
            feature_columns = disk_art.get("feature_columns", [])

        if pipeline is not None and lgb_model is not None:
            return {
                "model": lgb_model,
                "pipeline": pipeline,
                "metrics": metrics,
                "top_features": top_features,
                "feature_columns": feature_columns or pipeline.feature_columns,
                "model_version": version_str,
                "source": "mlflow_model_registry"
            }
    except Exception as e:
        logger.warning(f"Could not load champion model from MLflow registry: {e}. Falling back to disk artifact.")
    return None


def get_model() -> Optional[Dict[str, Any]]:
    """
    Non-raising model getter for healthchecks and lifecycle startup.
    Prioritizes loading directly from MLflow Model Registry, falling back to disk artifact.
    """
    global _cached_artifact
    if _cached_artifact is None:
        _cached_artifact = load_model_from_mlflow_registry()
        if _cached_artifact is None and os.path.exists(settings.MODEL_PATH):
            try:
                _cached_artifact = joblib.load(settings.MODEL_PATH)
                logger.info(f"Successfully loaded champion model fallback from {settings.MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not preload model artifact: {e}")
                return None
    return _cached_artifact


def get_model_artifact() -> Dict[str, Any]:
    """
    Dependency provider that loads and caches the champion model artifact.
    Prioritizes MLflow Model Registry with fallback to disk artifact.
    Raises HTTPException 503 if no model is found.
    """
    artifact = get_model()
    if artifact is None:
        logger.warning(f"Champion model not found in MLflow registry or at {settings.MODEL_PATH}")
        raise HTTPException(
            status_code=503,
            detail="Champion model artifact could not be loaded from MLflow or disk. Train the model first."
        )
    return artifact


def reload_model_artifact() -> Optional[Dict[str, Any]]:
    """Forces cache invalidation and reloads the model artifact."""
    global _cached_artifact
    _cached_artifact = None
    return get_model()


def get_settings():
    """Provides application configuration settings."""
    return settings


_last_db_check: float = 0.0
_cached_db_status: bool = False


def is_db_connected() -> bool:
    """Verifies active connectivity to PostgreSQL database with a 5-second cache."""
    global _last_db_check, _cached_db_status
    import time
    now = time.time()
    if now - _last_db_check < 5.0:
        return _cached_db_status

    _last_db_check = now
    if db_mod.SessionLocal is None:
        _cached_db_status = False
        return False

    db = db_mod.SessionLocal()
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        _cached_db_status = True
        return True
    except Exception:
        _cached_db_status = False
        return False
    finally:
        db.close()


def __getattr__(name: str):
    if name == "SessionLocal":
        return db_mod.SessionLocal
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

