import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# Human-readable labels and explanations for engineered and raw features
FEATURE_DESCRIPTIONS = {
    "Contract_Month-to-month": "Month-to-month contract (no long-term commitment)",
    "Contract_One year": "One-year contract commitment",
    "Contract_Two year": "Two-year long-term loyalty contract",
    "InternetService_Fiber optic": "Fiber optic high-speed subscription (higher cost tier)",
    "InternetService_DSL": "DSL internet service",
    "InternetService_No": "No internet service subscribed",
    "PaymentMethod_Electronic check": "Electronic check payment method (higher historic churn rate)",
    "PaymentMethod_Credit card (automatic)": "Automatic credit card billing (stable retention)",
    "PaymentMethod_Bank transfer (automatic)": "Automatic bank transfer billing",
    "PaymentMethod_Mailed check": "Mailed paper check payment",
    "OnlineSecurity_Yes": "Online security active",
    "OnlineSecurity_No": "No online security protection",
    "TechSupport_Yes": "Dedicated technical support active",
    "TechSupport_No": "Lack of technical support subscription",
    "PaperlessBilling_Yes": "Paperless e-billing active",
    "PaperlessBilling_No": "Traditional paper billing",
    "tenure": "Tenure with company (months)",
    "tenure_year": "Tenure in years",
    "MonthlyCharges": "Monthly recurring bill amount",
    "TotalCharges": "Cumulative lifetime charges",
    "num_services": "Total count of active add-on services",
    "avg_monthly_charges": "Average monthly spend ratio",
    "has_fiber_optic": "Fiber optic connection",
    "has_tech_support": "Technical support enabled",
    "is_month_to_month": "Month-to-month contract flag",
    "SeniorCitizen": "Senior citizen subscriber"
}

def compute_shap_explanations(
    booster,
    feature_df: pd.DataFrame,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Computes exact TreeSHAP attribution values using LightGBM's native C++ implementation
    via `predict(..., pred_contrib=True)`. Blazing fast with zero external dependencies.

    Returns a structured breakdown of:
      - top_risk_drivers: features increasing churn probability
      - top_protective_factors: features decreasing churn probability
      - raw_attributions: complete feature -> shap_value mapping
    """
    # pred_contrib=True returns (N, num_features + 1), last column is expected value
    contribs = booster.predict(feature_df, pred_contrib=True)
    feature_names = list(feature_df.columns)

    explanations = []
    for row_idx in range(len(feature_df)):
        row_contribs = contribs[row_idx, :-1] # exclude bias
        base_value = float(contribs[row_idx, -1])

        # Pair features with their SHAP contribution
        pairs = []
        for feat_name, val in zip(feature_names, row_contribs):
            val_float = float(val)
            human_desc = FEATURE_DESCRIPTIONS.get(
                feat_name,
                feat_name.replace("_", " ").title()
            )
            pairs.append({
                "feature": feat_name,
                "importance": round(abs(val_float), 4),
                "shap_value": round(val_float, 4),
                "direction": "increases_risk" if val_float > 0 else "decreases_risk",
                "description": human_desc
            })

        # Separate risk drivers (positive SHAP) and protective factors (negative SHAP)
        risk_drivers = sorted(
            [p for p in pairs if p["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True
        )[:top_n]

        protective = sorted(
            [p for p in pairs if p["shap_value"] < 0],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )[:top_n]

        # Simple string explanations for backward compatibility
        summary_strings = []
        for r in risk_drivers[:3]:
            summary_strings.append(f"{r['description']} (increases churn risk)")
        if not summary_strings:
            summary_strings.append("Standard loyalty profile with low risk indicators")

        explanations.append({
            "base_value": round(base_value, 4),
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective,
            "summary_explanations": summary_strings,
            "all_attributions": {p["feature"]: p["shap_value"] for p in pairs}
        })

    return explanations
