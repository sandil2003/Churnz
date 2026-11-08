import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.train import train_model

def main():
    print("--> Starting LightGBM champion training with leak-free splits & MLflow tracking...")
    metrics = train_model()
    print("[SUCCESS] Training pipeline finished successfully!")
    print(f"Champion Model Performance Metrics: {metrics}")

if __name__ == "__main__":
    main()
