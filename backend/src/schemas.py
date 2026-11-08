from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """
    Pydantic schema representing a single customer record
    based on the Telco Customer Churn dataset.
    """
    customer_id: Optional[str] = Field(default="CUST-0001", description="Unique identifier for the customer")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=100, description="Number of months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple lines subscription")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Customer's internet service provider")
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Whether paperless billing is enabled")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Payment method used")
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly")
    TotalCharges: Optional[float] = Field(default=None, description="The total amount charged to the customer")

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def parse_total_charges(cls, v, info):
        # In the Kaggle dataset, new customers with tenure=0 have TotalCharges as empty space ' '
        if v is None or v == "" or v == " ":
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class FeatureContribution(BaseModel):
    """Structured TreeSHAP feature contribution."""
    feature: str = Field(..., description="Feature name")
    attribution: float = Field(..., description="TreeSHAP marginal attribution in margin/log-odds space")
    direction: Literal["increases_risk", "decreases_risk"] = Field(..., description="Direction of risk impact")
    display_text: str = Field(..., description="Human-understandable risk explanation")


class PredictionResponse(BaseModel):
    """Output prediction contract returned to API clients."""
    customer_id: str
    prediction_id: Optional[str] = Field(default=None, description="Unique prediction UUID")
    churn_prediction: int = Field(..., description="1 for Churn, 0 for Retain")
    churn_probability: float = Field(..., description="Predicted probability of churning [0.0 - 1.0]")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Categorical risk tier")
    model_version: Optional[str] = Field(default=None, description="Champion model artifact version")
    risk_factors: List[FeatureContribution] = Field(default_factory=list, description="Structured TreeSHAP feature contributions")
    explanation: List[str] = Field(default_factory=list, description="Key driving risk factors (backward-compatible strings)")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema with KPI metrics."""
    predictions: List[PredictionResponse]
    total_processed: int
    churn_count: int
    churn_rate: float
    avg_probability: float = Field(default=0.0, description="Average predicted probability across batch")
    batch_latency_ms: float = Field(default=0.0, description="Batch execution latency in milliseconds")


class FeedbackInput(BaseModel):
    """Feedback input schema for recording actual churn ground truth."""
    customer_id: str
    actual_churn: int = Field(..., ge=0, le=1, description="1 if customer churned, 0 otherwise")
    prediction_id: Optional[str] = Field(default=None, description="Optional associated prediction UUID")


class HealthResponse(BaseModel):
    """Structured health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    database_connected: bool = False
    model_path: Optional[str] = None


class ModelStatsResponse(BaseModel):
    """Model performance metrics and feature importances."""
    model_name: Optional[str] = "LightGBM Classifier"
    model_version: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    features_count: int = 0


class PredictionHistoryItem(BaseModel):
    """Inference audit log record."""
    id: int
    prediction_id: Optional[str] = None
    customer_id: str
    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    created_at: Optional[str] = None
