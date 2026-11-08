"""
State-Aware Telco Feature Engineering Pipeline.
Fixes single-row categorical dummy drop, novel category handling, and schema alignment.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TelcoFeaturePipeline:
    """
    State-aware Pandas feature pipeline for Telco Customer Churn.
    Learns categorical levels and numerical medians during fit(), ensuring identical
    36-feature output representation for single-row and batch inference.
    """

    def __init__(self):
        self.feature_columns: List[str] = []
        self.feature_names_: List[str] = []
        self.categories_mapping: Dict[str, List[str]] = {}
        self.numerical_medians: Dict[str, float] = {}
        self.is_fitted_: bool = False

        self.categorical_columns = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        self.numerical_columns = [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            "tenure_year", "num_services", "has_fiber_optic", "has_tech_support",
            "is_month_to_month", "avg_monthly_charges"
        ]

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans dirty strings, handles missing TotalCharges, and ensures correct datatypes."""
        df = df.copy()

        # Ensure tenure is integer >= 0
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
        else:
            df["tenure"] = 0

        # Ensure MonthlyCharges is float >= 0.0
        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
        else:
            df["MonthlyCharges"] = 0.0

        # Handle Kaggle's blank strings ' ' and missing values in TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
            # If tenure is 0, TotalCharges is 0; otherwise impute MonthlyCharges * tenure if missing
            df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"]).fillna(0.0)
        else:
            df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).fillna(0.0)

        # Standardize SeniorCitizen to int (0 or 1)
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        else:
            df["SeniorCitizen"] = 0

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes domain-specific telecom features with safe series operations."""
        df = self.clean_raw_data(df)

        # 1. Tenure in completed years
        df["tenure_year"] = (df["tenure"] // 12).astype(int)

        # 2. Count of value-added services customer subscribes to
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        df["num_services"] = 0
        for col in service_cols:
            if col in df.columns:
                df["num_services"] += (df[col] == "Yes").astype(int)

        # 3. High churn risk behavioral indicators (safe column presence check)
        if "InternetService" in df.columns:
            df["has_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
        else:
            df["has_fiber_optic"] = 0

        if "TechSupport" in df.columns:
            df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
        else:
            df["has_tech_support"] = 0

        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        else:
            df["is_month_to_month"] = 0

        # 4. Financial ratios (safe against tenure = 0)
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)

        return df

    def fit(self, df: pd.DataFrame):
        """Learns feature columns, categorical mappings, and numerical medians from training data."""
        processed = self.engineer_features(df)

        # 1. Learn numerical medians for fallback imputation
        self.numerical_medians = {}
        for col in self.numerical_columns:
            if col in processed.columns:
                valid_vals = pd.to_numeric(processed[col], errors="coerce").dropna()
                self.numerical_medians[col] = float(valid_vals.median()) if not valid_vals.empty else 0.0
            else:
                self.numerical_medians[col] = 0.0

        # 2. Learn deterministic category levels for each categorical column
        self.categories_mapping = {}
        for col in self.categorical_columns:
            if col in processed.columns:
                cats = sorted(processed[col].dropna().astype(str).unique().tolist())
                self.categories_mapping[col] = cats
                processed[col] = pd.Categorical(processed[col], categories=cats)
            else:
                self.categories_mapping[col] = []
                processed[col] = pd.Categorical([None] * len(processed), categories=[])

        # 3. One-hot encode categorical features with drop_first=True
        encoded = pd.get_dummies(processed, columns=self.categorical_columns, drop_first=True, dtype=float)

        # Exclude metadata and target columns from feature set
        cols_to_drop = [
            col for col in ["customerID", "customer_id", "Churn", "churn", "ingested_at", "id"]
            if col in encoded.columns
        ]
        feature_df = encoded.drop(columns=cols_to_drop, errors="ignore")

        self.feature_columns = list(feature_df.columns)
        self.feature_names_ = list(feature_df.columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input records into the exact 36-feature schema learned during fit."""
        if not self.is_fitted_:
            raise ValueError("TelcoFeaturePipeline is not fitted yet. Call 'fit' or 'fit_transform' first.")

        processed = self.engineer_features(df)

        # 1. Apply numerical medians for missing numerical values
        for col, med_val in self.numerical_medians.items():
            if col in processed.columns:
                processed[col] = pd.to_numeric(processed[col], errors="coerce").fillna(med_val).astype(float)
            else:
                processed[col] = float(med_val)

        # 2. State-aware categorical casting using learned categories_mapping
        for col in self.categorical_columns:
            cats = self.categories_mapping.get(col, [])
            allowed_set = set(cats)
            if col in processed.columns:
                # Retain strings while converting missing values to None
                col_series = processed[col].astype(str).where(processed[col].notna(), None)
                # Unseen/novel categories become None (prevents future Pandas4Warning and ensures 0 in dummies)
                col_series = col_series.where(col_series.isin(allowed_set), None)
                processed[col] = pd.Categorical(col_series, categories=cats)
            else:
                # Entire column missing from input: generate all-NaN Categorical series
                processed[col] = pd.Categorical([None] * len(processed), categories=cats)

        # 3. One-hot encode with drop_first=True
        encoded = pd.get_dummies(processed, columns=self.categorical_columns, drop_first=True, dtype=float)

        # 4. Ensure all fitted feature columns exist (fill any remaining missing columns with 0.0)
        for col in self.feature_columns:
            if col not in encoded.columns:
                encoded[col] = 0.0

        # 5. Return strictly the fitted feature columns in exact order and float dtype
        return encoded[self.feature_columns].astype(float)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fits pipeline and returns transformed X and target y (if present)."""
        self.fit(df)
        X = self.transform(df)

        y = None
        churn_col = "Churn" if "Churn" in df.columns else ("churn" if "churn" in df.columns else None)
        if churn_col is not None:
            churn_series = df[churn_col].astype(str).str.strip()
            y = churn_series.map({
                "Yes": 1, "No": 0, "1": 1, "0": 0,
                "True": 1, "False": 0, "true": 1, "false": 0
            }).fillna(0).astype(int)
            y.name = "Churn"

        return X, y
