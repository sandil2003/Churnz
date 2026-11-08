import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.tune import run_tuning

def main():
    print("--> Starting Optuna hyperparameter optimization study...")
    best_params = run_tuning(n_trials=15)
    print("[SUCCESS] Tuning complete and saved to backend/models/best_hyperparams.json!")
    print(f"Optimal parameters: {best_params}")

if __name__ == "__main__":
    main()
