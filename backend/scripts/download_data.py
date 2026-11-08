import os
import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_telco_data():
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEST_PATH.exists():
        print(f"[OK] Dataset already present at: {DEST_PATH}")
        return

    print(f"--> Downloading Telco Customer Churn dataset from verified source...")
    try:
        urllib.request.urlretrieve(DATA_URL, str(DEST_PATH))
        print(f"[SUCCESS] Downloaded to: {DEST_PATH}")
    except Exception as e:
        print(f"[ERROR] Automated download failed: {e}")
        print("Please manually download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"and place it in: {DEST_PATH}")

if __name__ == "__main__":
    download_telco_data()
