import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.monitoring import generate_drift_report, load_reference_data

def simulate_current_data_with_drift(reference_df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """
    Simulates a production month with subtle feature drift
    (e.g. inflation causing higher MonthlyCharges and higher month-to-month contracts).
    """
    sample = reference_df.sample(n=min(sample_size, len(reference_df)), replace=False, random_state=123).copy()

    # Induce drift in MonthlyCharges (shift mean up by $15)
    if "MonthlyCharges" in sample.columns:
        sample["MonthlyCharges"] = sample["MonthlyCharges"] + np.random.normal(15.0, 3.0, size=len(sample))

    # Induce drift in tenure (influx of new customers)
    if "tenure" in sample.columns:
        sample["tenure"] = np.maximum(0, sample["tenure"] - 10)

    return sample

def main():
    print("--> Checking for reference training data...")
    ref_df = load_reference_data()

    if ref_df is None:
        print("[ERROR] Reference data not found! Please train the model first by running:")
        print("python scripts/run_training.py")
        return

    print(f"--> Loaded reference dataset with {len(ref_df)} records.")
    print("--> Simulating recent production cohort to test data drift...")
    current_df = simulate_current_data_with_drift(ref_df)

    summary = generate_drift_report(
        reference_df=ref_df,
        current_df=current_df,
        report_filename="drift_report.html"
    )

    print("\n[SUCCESS] Evidently AI Drift Report Generated!")
    print(f"Interactive Report saved to: {os.path.join(settings.REPORTS_PATH, 'drift_report.html')}")
    print("Open the file in your browser or view it via GET http://localhost:8000/reports/drift\n")

if __name__ == "__main__":
    main()
