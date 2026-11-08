import io
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File

from src.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureContribution
)
from src.api.deps import get_model_artifact
from src.database import log_prediction_to_db, log_batch_predictions_to_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


def format_feature_explanation(
    feature: str,
    attribution: float,
    raw_record: Optional[Dict[str, Any]] = None
) -> str:
    """Formats a human-readable explanation for a specific feature contribution."""
    raw = raw_record if isinstance(raw_record, dict) else {}
    attr_positive = attribution > 0

    if feature in ("is_month_to_month", "Contract") or feature.startswith("Contract_"):
        if feature == "Contract_Two year":
            return (
                "Two-year contract provides exceptional retention stability"
                if not attr_positive
                else "Two-year contract approaching expiration window"
            )
        if feature == "Contract_One year":
            return (
                "One-year annual contract reduces churn risk"
                if not attr_positive
                else "One-year annual contract approaching renewal window"
            )
        return (
            "Month-to-month contract significantly elevates cancellation risk"
            if attr_positive
            else "Long-term contract structure strongly encourages retention"
        )

    if feature == "tenure":
        t = raw.get("tenure")
        if t is not None:
            return (
                f"Short tenure ({t} months) creates new subscriber churn vulnerability"
                if attr_positive
                else f"Established tenure ({t} months) promotes customer loyalty"
            )
        return (
            "Short tenure creates new subscriber churn vulnerability"
            if attr_positive
            else "Established tenure promotes customer loyalty"
        )

    if feature == "tenure_year":
        return (
            "First-year tenure creates new subscriber vulnerability"
            if attr_positive
            else "Multi-year subscription history reinforces loyalty"
        )

    if feature in ("has_fiber_optic", "InternetService_Fiber optic"):
        return (
            "Fiber optic service plan associated with higher competitive churn"
            if attr_positive
            else "Non-fiber internet tier aligns with customer retention"
        )

    if feature == "InternetService_No":
        return (
            "Absence of broadband services correlates with service termination"
            if attr_positive
            else "No internet service baseline profile exhibits very low churn"
        )

    if feature == "PaymentMethod_Electronic check":
        return (
            "Electronic check payment method statistically correlates with higher churn"
            if attr_positive
            else "Electronic check profile remains within normal retention bounds"
        )

    if feature in ("PaymentMethod_Credit card (automatic)", "PaymentMethod_Bank transfer (automatic)"):
        return (
            "Manual payment intervention required"
            if attr_positive
            else "Automated payment method ensures continuous billing and reduces churn"
        )

    if feature in ("has_tech_support", "TechSupport_Yes"):
        return (
            "Tech Support friction indicates customer dissatisfaction"
            if attr_positive
            else "Active Tech Support coverage substantially decreases churn probability"
        )

    if feature == "TechSupport_No internet service":
        return "No internet service baseline lowers support churn risk"

    if feature == "OnlineSecurity_Yes":
        return (
            "Security add-on adjustment increases churn sensitivity"
            if attr_positive
            else "Online Security add-on creates product stickiness and reduces churn"
        )

    if feature == "OnlineBackup_Yes":
        return (
            "Backup service adjustment indicates plan re-evaluation"
            if attr_positive
            else "Online Backup add-on creates high switching costs and reduces churn"
        )

    if feature == "DeviceProtection_Yes":
        return (
            "Device protection re-evaluation indicates potential downgrade"
            if attr_positive
            else "Device Protection plan promotes long-term retention"
        )

    if feature == "PaperlessBilling_Yes":
        return (
            "Paperless billing correlates with price sensitivity and higher churn"
            if attr_positive
            else "Standard paper billing indicates established customer habit"
        )

    if feature == "MonthlyCharges":
        mc = raw.get("MonthlyCharges")
        try:
            mc_val = float(mc) if mc is not None else None
            if mc_val is not None and (np.isnan(mc_val) or np.isinf(mc_val)):
                mc_val = None
        except (ValueError, TypeError):
            mc_val = None
        if mc_val is not None:
            return (
                f"High monthly charges (${mc_val:.2f}/mo) increase price sensitivity and churn risk"
                if attr_positive
                else f"Affordable monthly charges (${mc_val:.2f}/mo) protect against churn"
            )
        return (
            "High monthly charges increase price sensitivity and churn risk"
            if attr_positive
            else "Affordable monthly charges protect against churn"
        )

    if feature == "TotalCharges":
        tc = raw.get("TotalCharges")
        try:
            tc_val = float(tc) if tc is not None else None
            if tc_val is not None and (np.isnan(tc_val) or np.isinf(tc_val)):
                tc_val = None
        except (ValueError, TypeError):
            tc_val = None
        if tc_val is not None:
            return (
                f"Accumulated spend (${tc_val:.2f}) indicates elevated churn sensitivity"
                if attr_positive
                else f"High lifetime value (${tc_val:.2f}) reflects significant customer commitment"
            )
        return (
            "Accumulated total spend indicates elevated churn sensitivity"
            if attr_positive
            else "High lifetime value reflects customer commitment"
        )

    if feature == "avg_monthly_charges":
        return (
            "Average monthly charge trajectory indicates recent cost escalation"
            if attr_positive
            else "Average monthly charges indicate consistent billing predictability"
        )

    if feature == "num_services":
        return (
            "Limited service bundle adoption exhibits lower switching barriers"
            if attr_positive
            else "High service bundle adoption deepens relationship and reduces churn"
        )

    # General fallback
    readable = feature.replace("_", " ")
    direction_text = "increases churn risk" if attr_positive else "decreases churn risk"
    return f"{readable}: {direction_text} ({attribution:+.3f})"


def compute_treeshap_contributions(
    model: Any,
    pipeline: Any,
    X: np.ndarray,
    raw_record: Optional[Dict[str, Any]] = None,
    top_pos: int = 3,
    top_neg: int = 2,
    feat_contribs: Optional[np.ndarray] = None
) -> tuple[List[FeatureContribution], List[str]]:
    """
    Computes authentic LightGBM TreeSHAP feature attributions using
    model.booster_.predict(X, pred_contrib=True).
    Extracts top positive (risk-increasing) and top negative (risk-decreasing) features.
    """
    feature_names = getattr(pipeline, "feature_columns", [f"feat_{i}" for i in range(X.shape[1])])

    if feat_contribs is None:
        if hasattr(model, "booster_"):
            contribs = model.booster_.predict(X, pred_contrib=True)
            feat_contribs = contribs[0, :-1]
        else:
            # Fallback if mock model without booster_
            if hasattr(model, "feature_importances_"):
                feat_contribs = model.feature_importances_.astype(float)
            else:
                feat_contribs = np.zeros(len(feature_names))

    pos_indices = [idx for idx in np.argsort(feat_contribs)[::-1] if feat_contribs[idx] > 0][:top_pos]
    neg_indices = [idx for idx in np.argsort(feat_contribs) if feat_contribs[idx] < 0][:top_neg]

    selected_indices = pos_indices + neg_indices
    if not selected_indices:
        selected_indices = list(np.argsort(np.abs(feat_contribs))[::-1][:top_pos + top_neg])

    risk_factors: List[FeatureContribution] = []
    explanations: List[str] = []

    for idx in selected_indices:
        feat = feature_names[idx]
        val = float(feat_contribs[idx])
        direction = "increases_risk" if val > 0 else "decreases_risk"
        display = format_feature_explanation(feat, val, raw_record)

        risk_factors.append(FeatureContribution(
            feature=feat,
            attribution=round(val, 4),
            direction=direction,
            display_text=display
        ))
        explanations.append(display)

    if not explanations:
        explanations.append("Standard loyalty profile with low risk indicators")

    return risk_factors, explanations


def assign_risk_level(prob: float) -> str:
    """Categorizes churn probability into risk tier."""
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


@router.post("/predict", response_model=PredictionResponse, summary="Predict Single Customer Churn")
def predict_churn(
    customer: CustomerInput,
    background_tasks: BackgroundTasks,
    artifact: Dict[str, Any] = Depends(get_model_artifact)
):
    """
    Scores a single customer record and returns churn probability, risk level,
    and authentic LightGBM TreeSHAP feature attributions.
    Logs inference event to PostgreSQL in background.
    """
    start_time = time.perf_counter()
    model = artifact["model"]
    pipeline = artifact["pipeline"]
    model_version = artifact.get("model_version", "champion-v1.0.0")

    input_dict = customer.model_dump()
    df_raw = pd.DataFrame([input_dict])

    try:
        X = pipeline.transform(df_raw)
    except Exception as e:
        logger.error(f"Pipeline transformation error: {e}")
        raise HTTPException(status_code=422, detail=f"Feature transformation failed: {e}")

    try:
        churn_prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")

    churn_pred = int(churn_prob >= 0.5)
    risk_level = assign_risk_level(churn_prob)
    prediction_id = str(uuid.uuid4())

    risk_factors, explanations = compute_treeshap_contributions(
        model=model,
        pipeline=pipeline,
        X=X,
        raw_record=input_dict
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Persist inference audit event to PostgreSQL via BackgroundTasks
    background_tasks.add_task(
        log_prediction_to_db,
        customer_dict=input_dict,
        prediction=churn_pred,
        probability=churn_prob,
        risk_level=risk_level,
        prediction_id=prediction_id,
        model_version=model_version,
        latency_ms=latency_ms
    )

    return PredictionResponse(
        customer_id=customer.customer_id or "CUST-0001",
        prediction_id=prediction_id,
        churn_prediction=churn_pred,
        churn_probability=round(churn_prob, 4),
        risk_level=risk_level,
        model_version=model_version,
        risk_factors=risk_factors,
        explanation=explanations
    )


@router.post("/batch-predict", response_model=BatchPredictionResponse, summary="Batch Inference (JSON or CSV)")
async def batch_predict(
    request: Request,
    background_tasks: BackgroundTasks,
    artifact: Dict[str, Any] = Depends(get_model_artifact)
):
    """
    Performs vectorized batch prediction from JSON payload or CSV file upload.
    Returns summary KPI metrics (total processed, churn count, churn rate, avg probability, latency)
    and detailed predictions. Logs batch predictions to PostgreSQL in background.
    """
    start_time = time.perf_counter()
    model = artifact["model"]
    pipeline = artifact["pipeline"]
    model_version = artifact.get("model_version", "champion-v1.0.0")

    content_type = request.headers.get("content-type", "")

    # Parse input into records
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file or not hasattr(uploaded_file, "read"):
            raise HTTPException(status_code=422, detail="No CSV file uploaded under key 'file'.")
        contents = await uploaded_file.read()
        try:
            df_raw = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse uploaded CSV: {e}")

        # Fill missing customer_id if absent
        if "customer_id" not in df_raw.columns:
            df_raw["customer_id"] = [f"BATCH-{i+1:04d}" for i in range(len(df_raw))]

        records = df_raw.to_dict(orient="records")
    else:
        try:
            json_body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {e}")

        if isinstance(json_body, list):
            records = json_body
        elif isinstance(json_body, dict) and "customers" in json_body:
            records = json_body["customers"]
        else:
            raise HTTPException(
                status_code=422,
                detail="Expected JSON body with 'customers' array or list of customer objects."
            )

    if not records:
        raise HTTPException(status_code=422, detail="Batch customer list cannot be empty.")

    # Validate with CustomerInput schema
    validated_records = []
    for i, r in enumerate(records):
        try:
            c = CustomerInput(**r)
            validated_records.append(c.model_dump())
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Row {i} validation failed: {e}")
    records = validated_records
    df_raw = pd.DataFrame(records)

    if len(df_raw) == 0:
        raise HTTPException(status_code=422, detail="No customer records found to process.")

    try:
        X = pipeline.transform(df_raw)
    except Exception as e:
        logger.error(f"Batch transformation error: {e}")
        raise HTTPException(status_code=422, detail=f"Batch feature transformation failed: {e}")

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {e}")

    preds = (probs >= 0.5).astype(int)

    # Compute TreeSHAP contributions once across the entire batch matrix
    if hasattr(model, "booster_"):
        batch_contribs = model.booster_.predict(X, pred_contrib=True)
    else:
        batch_contribs = None

    results: List[PredictionResponse] = []
    log_records: List[Dict[str, Any]] = []
    churn_count = 0

    for i, row in enumerate(records):
        p = float(probs[i])
        pred = int(preds[i])
        if pred == 1:
            churn_count += 1

        tier = assign_risk_level(p)
        pred_id = str(uuid.uuid4())
        cid = str(row.get("customer_id") or f"CUST-{i+1:04d}")

        # Compute individual row TreeSHAP from precomputed batch matrix contributions
        row_feat_contribs = batch_contribs[i, :-1] if batch_contribs is not None else None
        risk_factors, explanations = compute_treeshap_contributions(
            model=model,
            pipeline=pipeline,
            X=X[i:i+1],
            raw_record=row,
            feat_contribs=row_feat_contribs
        )

        results.append(PredictionResponse(
            customer_id=cid,
            prediction_id=pred_id,
            churn_prediction=pred,
            churn_probability=round(p, 4),
            risk_level=tier,
            model_version=model_version,
            risk_factors=risk_factors,
            explanation=explanations
        ))

        log_records.append({
            "features": row,
            "customer_id": cid,
            "prediction_id": pred_id,
            "churn_prediction": pred,
            "churn_probability": p,
            "risk_level": tier,
            "model_version": model_version
        })

    total_processed = len(records)
    churn_rate = round(churn_count / max(total_processed, 1), 4)
    avg_probability = round(float(np.mean(probs)), 4)
    batch_latency = round((time.perf_counter() - start_time) * 1000, 2)

    # Enqueue batch database logging via BackgroundTasks
    background_tasks.add_task(
        log_batch_predictions_to_db,
        records=log_records,
        model_version=model_version
    )

    return BatchPredictionResponse(
        predictions=results,
        total_processed=total_processed,
        churn_count=churn_count,
        churn_rate=churn_rate,
        avg_probability=avg_probability,
        batch_latency_ms=batch_latency
    )


@router.post("/batch-predict/csv", response_model=BatchPredictionResponse, summary="Batch Inference via CSV File")
async def batch_predict_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    artifact: Dict[str, Any] = Depends(get_model_artifact)
):
    """Explicit multipart CSV upload endpoint for batch prediction workflows."""
    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV file: {e}")

    if "customer_id" not in df_raw.columns:
        df_raw["customer_id"] = [f"BATCH-{i+1:04d}" for i in range(len(df_raw))]

    records = df_raw.to_dict(orient="records")
    validated_records = []
    for i, r in enumerate(records):
        try:
            c = CustomerInput(**r)
            validated_records.append(c.model_dump())
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Row {i} invalid: {e}")

    if not records or not validated_records:
        raise HTTPException(status_code=422, detail="Batch customer list cannot be empty.")

    start_time = time.perf_counter()
    model = artifact["model"]
    pipeline = artifact["pipeline"]
    model_version = artifact.get("model_version", "champion-v1.0.0")

    try:
        X = pipeline.transform(pd.DataFrame(validated_records))
    except Exception as e:
        logger.error(f"Batch transformation error: {e}")
        raise HTTPException(status_code=422, detail=f"Batch feature transformation failed: {e}")

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {e}")

    preds = (probs >= 0.5).astype(int)

    # Compute TreeSHAP contributions once across the entire batch matrix
    if hasattr(model, "booster_"):
        batch_contribs = model.booster_.predict(X, pred_contrib=True)
    else:
        batch_contribs = None

    results = []
    log_records = []
    churn_count = 0

    for i, row in enumerate(validated_records):
        p = float(probs[i])
        pred = int(preds[i])
        if pred == 1:
            churn_count += 1

        tier = assign_risk_level(p)
        pred_id = str(uuid.uuid4())
        cid = str(row.get("customer_id") or f"CUST-{i+1:04d}")

        row_feat_contribs = batch_contribs[i, :-1] if batch_contribs is not None else None
        risk_factors, explanations = compute_treeshap_contributions(
            model=model,
            pipeline=pipeline,
            X=X[i:i+1],
            raw_record=row,
            feat_contribs=row_feat_contribs
        )

        results.append(PredictionResponse(
            customer_id=cid,
            prediction_id=pred_id,
            churn_prediction=pred,
            churn_probability=round(p, 4),
            risk_level=tier,
            model_version=model_version,
            risk_factors=risk_factors,
            explanation=explanations
        ))

        log_records.append({
            "features": row,
            "customer_id": cid,
            "prediction_id": pred_id,
            "churn_prediction": pred,
            "churn_probability": p,
            "risk_level": tier,
            "model_version": model_version
        })

    total_processed = len(validated_records)
    churn_rate = round(churn_count / max(total_processed, 1), 4)
    avg_probability = round(float(np.mean(probs)), 4)
    batch_latency = round((time.perf_counter() - start_time) * 1000, 2)

    background_tasks.add_task(
        log_batch_predictions_to_db,
        records=log_records,
        model_version=model_version
    )

    return BatchPredictionResponse(
        predictions=results,
        total_processed=total_processed,
        churn_count=churn_count,
        churn_rate=churn_rate,
        avg_probability=avg_probability,
        batch_latency_ms=batch_latency
    )
