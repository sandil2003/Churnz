"""Initial database schema with hardened audit logs

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-11-07 13:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. raw_customers
    op.create_table(
        "raw_customers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.String(length=50), nullable=False),
        sa.Column("gender", sa.String(length=20), nullable=True),
        sa.Column("SeniorCitizen", sa.Integer(), nullable=True),
        sa.Column("Partner", sa.String(length=10), nullable=True),
        sa.Column("Dependents", sa.String(length=10), nullable=True),
        sa.Column("tenure", sa.Integer(), nullable=True),
        sa.Column("PhoneService", sa.String(length=10), nullable=True),
        sa.Column("MultipleLines", sa.String(length=30), nullable=True),
        sa.Column("InternetService", sa.String(length=30), nullable=True),
        sa.Column("OnlineSecurity", sa.String(length=30), nullable=True),
        sa.Column("OnlineBackup", sa.String(length=30), nullable=True),
        sa.Column("DeviceProtection", sa.String(length=30), nullable=True),
        sa.Column("TechSupport", sa.String(length=30), nullable=True),
        sa.Column("StreamingTV", sa.String(length=30), nullable=True),
        sa.Column("StreamingMovies", sa.String(length=30), nullable=True),
        sa.Column("Contract", sa.String(length=30), nullable=True),
        sa.Column("PaperlessBilling", sa.String(length=10), nullable=True),
        sa.Column("PaymentMethod", sa.String(length=50), nullable=True),
        sa.Column("MonthlyCharges", sa.Float(), nullable=True),
        sa.Column("TotalCharges", sa.Float(), nullable=True),
        sa.Column("Churn", sa.String(length=10), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id")
    )
    op.create_index(op.f("ix_raw_customers_customer_id"), "raw_customers", ["customer_id"], unique=True)

    # 2. prediction_logs
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("customer_id", sa.String(length=50), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("churn_prediction", sa.Integer(), nullable=False),
        sa.Column("churn_probability", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id")
    )
    op.create_index(op.f("ix_prediction_logs_prediction_id"), "prediction_logs", ["prediction_id"], unique=True)
    op.create_index(op.f("ix_prediction_logs_customer_id"), "prediction_logs", ["customer_id"], unique=False)
    op.create_index(op.f("ix_prediction_logs_model_version"), "prediction_logs", ["model_version"], unique=False)
    op.create_index(op.f("ix_prediction_logs_created_at"), "prediction_logs", ["created_at"], unique=False)

    # 3. ground_truth_feedback
    op.create_table(
        "ground_truth_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.String(length=50), nullable=False),
        sa.Column("actual_churn", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id")
    )
    op.create_index(op.f("ix_ground_truth_feedback_customer_id"), "ground_truth_feedback", ["customer_id"], unique=False)
    op.create_index(op.f("ix_ground_truth_feedback_prediction_id"), "ground_truth_feedback", ["prediction_id"], unique=False)

def downgrade() -> None:
    op.drop_table("ground_truth_feedback")
    op.drop_table("prediction_logs")
    op.drop_table("raw_customers")
