import logging
from fastapi import APIRouter, Depends
from src.schemas import FeedbackInput
from src.api.deps import get_db
from src.database import GroundTruthTable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback"])


@router.post("/feedback", summary="Submit Ground Truth Outcome")
def submit_feedback(feedback: FeedbackInput, db=Depends(get_db)):
    """
    Records verified customer churn ground truth for monitoring and retraining feedback loops.
    Gracefully handles database offline mode without throwing 500 errors.
    """
    if db is None:
        logger.info(f"Feedback received for customer {feedback.customer_id} (DB offline mode)")
        return {
            "status": "ok",
            "message": "Feedback received (DB offline)",
            "customer_id": feedback.customer_id,
            "actual_churn": feedback.actual_churn
        }

    try:
        record = GroundTruthTable(
            customer_id=feedback.customer_id,
            actual_churn=feedback.actual_churn,
            prediction_id=feedback.prediction_id
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"Recorded ground truth outcome for customer {feedback.customer_id}: churn={feedback.actual_churn}")
        return {
            "status": "success",
            "id": record.id,
            "customer_id": feedback.customer_id,
            "actual_churn": feedback.actual_churn,
            "prediction_id": feedback.prediction_id
        }
    except Exception as e:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        logger.warning(f"Database error while recording feedback for customer {feedback.customer_id}: {e}")
        return {
            "status": "ok",
            "message": f"Feedback received (DB offline or error: {e})",
            "customer_id": feedback.customer_id,
            "actual_churn": feedback.actual_churn
        }
