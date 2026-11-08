from fastapi import APIRouter
from src.api.health import router as health_router
from src.api.predict import router as predict_router
from src.api.feedback import router as feedback_router
from src.api.monitoring import router as monitoring_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(feedback_router)
api_router.include_router(monitoring_router)

__all__ = ["api_router"]
