"""
MediAssist - Setup Verification Script
========================================
Run this script after cloning the repository and installing dependencies
to confirm that every component is working before doing any development.

Usage
-----
    python verify_setup.py

Exit codes
----------
    0  All checks passed — environment is ready.
    1  One or more checks failed — see output for details.
"""

import sys
import importlib
import json


REQUIRED_PACKAGES = [
    "pandas", "numpy", "sklearn", "matplotlib",
    "seaborn", "joblib", "lightgbm", "xgboost", "streamlit",
]

REQUIRED_ARTIFACT_FILES = [
    "models/final_model.pkl",
    "models/scaler.pkl",
    "models/feature_names.pkl",
    "models/threshold.pkl",
]

REQUIRED_DATASET = "data/cardio_train.csv"

BRIDGE_TEST_PATIENT = {
    "age": 58, "gender": 2, "height": 170, "weight": 92,
    "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 0,
}

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(label: str, fn) -> bool:
    """Run `fn`, print pass/fail, return True on success."""
    try:
        result = fn()
        msg = f" — {result}" if result else ""
        print(f"  {PASS} {label}{msg}")
        return True
    except Exception as exc:
        print(f"  {FAIL} {label}")
        print(f"         {exc}")
        return False


def main() -> int:
    print("\nMediAssist — Setup Verification\n" + "=" * 50)

    failures = 0

    # ------------------------------------------------------------------
    # 1. Python version
    # ------------------------------------------------------------------
    print("\n[1] Python version")
    major, minor = sys.version_info[:2]
    ok = check(
        f"Python {major}.{minor} (requires 3.10+)",
        lambda: None if (major, minor) >= (3, 10)
        else (_ for _ in ()).throw(RuntimeError(f"Found {major}.{minor}, need ≥3.10")),
    )
    failures += not ok

    # ------------------------------------------------------------------
    # 2. Package imports
    # ------------------------------------------------------------------
    print("\n[2] Required packages")
    for pkg in REQUIRED_PACKAGES:
        ok = check(f"import {pkg}", lambda p=pkg: importlib.import_module(p).__version__)
        failures += not ok

    # ------------------------------------------------------------------
    # 3. Dataset file
    # ------------------------------------------------------------------
    print("\n[3] Dataset")
    from pathlib import Path

    def _check_dataset():
        p = Path(REQUIRED_DATASET)
        if not p.exists():
            raise FileNotFoundError(
                f"{REQUIRED_DATASET} not found.\n"
                "         Download from https://www.kaggle.com/datasets/sulianova/cardiovascular-disease\n"
                "         and place cardio_train.csv in the data/ folder, then run:  python main.py"
            )
        import pandas as pd
        df = pd.read_csv(p, sep=";", nrows=5)
        return f"{p} ({len(df.columns)} columns, delimiter=';')"

    ok = check(f"{REQUIRED_DATASET} readable", _check_dataset)
    failures += not ok

    # ------------------------------------------------------------------
    # 4. Model artifacts
    # ------------------------------------------------------------------
    print("\n[4] Model artifacts  (run 'python main.py' if any are missing)")
    for path_str in REQUIRED_ARTIFACT_FILES:
        p = Path(path_str)
        ok = check(
            path_str,
            lambda p=p: None if p.exists()
            else (_ for _ in ()).throw(FileNotFoundError(f"Not found — run python main.py")),
        )
        failures += not ok

    # ------------------------------------------------------------------
    # 5. bridge.py end-to-end
    # ------------------------------------------------------------------
    print("\n[5] bridge.py end-to-end test")

    def _bridge_test():
        from bridge import get_prediction
        result = get_prediction(BRIDGE_TEST_PATIENT)
        assert "prediction" in result, "Missing 'prediction' key"
        assert "probability" in result, "Missing 'probability' key"
        assert "threshold_used" in result, "Missing 'threshold_used' key"
        assert "risk_assessment" in result, "Missing 'risk_assessment' key"
        assert result["prediction"] in (0, 1), "prediction must be 0 or 1"
        assert 0.0 <= result["probability"] <= 1.0, "probability out of range"
        return (
            f"prediction={result['prediction']}, "
            f"probability={result['probability']:.4f}, "
            f"threshold={result['threshold_used']:.4f}, "
            f"risk_level={result['risk_assessment']['risk_level']}"
        )

    ok = check("get_prediction() returns valid response", _bridge_test)
    failures += not ok

    # ------------------------------------------------------------------
    # 6. Knowledge engine
    # ------------------------------------------------------------------
    print("\n[6] Knowledge engine")

    def _ke_test():
        from src.knowledge_engine import assess_risk
        bmi = BRIDGE_TEST_PATIENT["weight"] / (BRIDGE_TEST_PATIENT["height"] / 100) ** 2
        r = assess_risk({**BRIDGE_TEST_PATIENT, "bmi": round(bmi, 2)})
        assert r["risk_level"] in ("High Risk", "Moderate Risk", "Low Risk")
        return f"risk_level={r['risk_level']}, flags={r['flag_count']}"

    ok = check("assess_risk() runs without error", _ke_test)
    failures += not ok

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    if failures == 0:
        print(f"  {PASS} All checks passed. Environment is ready.\n")
        print("  Next steps:")
        print("    • Run the training pipeline:   python main.py")
        print("    • Launch the demo interface:   streamlit run app.py")
        print("    • Read the API docs:           README.md")
        print("    • Read methodology & metrics:  PROJECT_LOG.md\n")
    else:
        print(f"  {FAIL} {failures} check(s) failed. Fix the issues above.\n")
        print("  Common fixes:")
        print("    • Missing packages:   pip install -r requirements.txt")
        print("    • Missing artifacts:  python main.py")
        print("    • Missing dataset:    download from Kaggle → data/cardio_train.csv\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
