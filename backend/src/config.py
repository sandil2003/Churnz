import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application and infrastructure configuration settings."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # PostgreSQL Database URL with explicit psycopg2 driver
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/churn_db"
    
    # MLflow Tracking URI
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_EXPERIMENT_NAME: str = "telco_customer_churn"
    MLFLOW_MODEL_NAME: str = "TelcoChurnChampion"
    MLFLOW_MODEL_ALIAS: str = "champion"
    
    # Paths
    DATA_RAW_PATH: str = str(BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    DATA_PROCESSED_PATH: str = str(BASE_DIR / "data" / "processed")
    MODEL_PATH: str = str(BASE_DIR / "models" / "champion_model.joblib")
    HYPERPARAMS_PATH: str = str(BASE_DIR / "models" / "best_hyperparams.json")
    REPORTS_PATH: str = str(BASE_DIR / "reports")

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def assemble_db_connection(cls, v: str) -> str:
        """
        Ensures the PostgreSQL connection string specifies the psycopg2 driver.
        Upgrades 'postgres://' or driverless 'postgresql://' to 'postgresql+psycopg2://'
        while preserving other schemes (e.g. 'sqlite:///').
        """
        if not v:
            return v
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg2://", 1)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg2://", 1)
        return v

settings = Settings()
