import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
import src.database as db_mod
from src.database import CustomerTable

def ingest_csv_to_db(csv_path: str = settings.DATA_RAW_PATH):
    print("--> Connecting to PostgreSQL database...")
    if not db_mod.init_db_connection() or db_mod.SessionLocal is None:
        print("[ERROR] Cannot connect to PostgreSQL. Run 'docker compose up -d postgres' first.")
        return

    print(f"--> Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean TotalCharges empty spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    db = db_mod.SessionLocal()
    try:
        existing_count = db.query(CustomerTable).count()
        if existing_count > 0:
            print(f"[INFO] Database already contains {existing_count} records. Skipping duplicate ingestion.")
            return

        print(f"--> Ingesting {len(df)} customer records into 'raw_customers' table...")
        records = []
        for _, row in df.iterrows():
            record = CustomerTable(
                customer_id=row["customerID"],
                gender=row["gender"],
                SeniorCitizen=int(row["SeniorCitizen"]),
                Partner=row["Partner"],
                Dependents=row["Dependents"],
                tenure=int(row["tenure"]),
                PhoneService=row["PhoneService"],
                MultipleLines=row["MultipleLines"],
                InternetService=row["InternetService"],
                OnlineSecurity=row["OnlineSecurity"],
                OnlineBackup=row["OnlineBackup"],
                DeviceProtection=row["DeviceProtection"],
                TechSupport=row["TechSupport"],
                StreamingTV=row["StreamingTV"],
                StreamingMovies=row["StreamingMovies"],
                Contract=row["Contract"],
                PaperlessBilling=row["PaperlessBilling"],
                PaymentMethod=row["PaymentMethod"],
                MonthlyCharges=float(row["MonthlyCharges"]),
                TotalCharges=float(row["TotalCharges"]),
                Churn=row.get("Churn")
            )
            records.append(record)

        db.bulk_save_objects(records)
        db.commit()
        print(f"[SUCCESS] Ingested {len(records)} records into PostgreSQL!")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Ingestion failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    ingest_csv_to_db()
