# Contributing to MediAssist

Welcome. This document explains the project structure, how to get set up, and the conventions to follow when working on the codebase.

---

## Project Overview

MediAssist is a cardiovascular disease risk assessment backend. It uses a hybrid approach:

1. **Machine learning pipeline** — three trained models (Logistic Regression, Random Forest, LightGBM) selected by Recall and calibrated with a tuned probability threshold
2. **Rule-based knowledge engine** — evidence-based clinical heuristics that flag specific risk factors independently of the ML model

The two layers are combined in `bridge.py`, which is the **only file a frontend needs to import**.

---

## Architecture

```
Training (run once)         Inference (per request)
─────────────────────       ────────────────────────────
main.py                     bridge.get_prediction(patient_dict)
  │                           │
  ├── src/data_pipeline.py    ├── bridge._validate_input()
  ├── src/ml_models.py        ├── bridge._preprocess()   ← computes BMI, pulse_pressure, age_bmi
  ├── src/evaluation.py       ├── model.predict_proba()  ← loaded from models/final_model.pkl
  └── src/knowledge_engine.py └── knowledge_engine.assess_risk()
         │
         └── models/*.pkl     ← consumed at inference
```

---

## Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/Ramendan/MediAssist.git
cd MediAssist

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download cardio_train.csv from:
#   https://www.kaggle.com/datasets/sulianova/cardiovascular-disease
# Place it at:  data/cardio_train.csv

# 5. Run setup verification
python verify_setup.py

# 6. Train the model (generates models/*.pkl and plots/*.png)
python main.py

# 7. Launch the demo UI  (optional)
streamlit run app.py
```

---

## File Responsibilities

| File | Owner | Purpose |
|------|-------|---------|
| `main.py` | Backend | Training pipeline orchestrator — run to regenerate all artifacts |
| `bridge.py` | Backend | Inference interface — the only file frontend imports |
| `app.py` | Reference | Temporary Streamlit demo / frontend reference implementation |
| `config.py` | Backend | Central configuration (split ratios, hyperparameter settings, paths) |
| `verify_setup.py` | Both | Environment health check — run after cloning |
| `src/data_pipeline.py` | Backend | Load, clean, feature engineer, split, normalise |
| `src/ml_models.py` | Backend | Train, tune, select, threshold optimisation |
| `src/evaluation.py` | Backend | Generate all 5 evaluation plots |
| `src/knowledge_engine.py` | Backend | Rule-based risk assessment |

---

## Regenerating the Model

Any time you change preprocessing, features, or model hyperparameters, retrain:

```bash
python -W ignore main.py
```

This overwrites `models/*.pkl`, `plots/*.png`, and `PROJECT_LOG.md`.

---

## Frontend Integration

The entire frontend API is one function:

```python
from bridge import get_prediction

result = get_prediction({
    "age": 45, "gender": 1, "height": 165, "weight": 68,
    "ap_hi": 125, "ap_lo": 82, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1,
})
# result = {
#   "prediction": 0 | 1,
#   "probability": 0.0–1.0,
#   "threshold_used": float,
#   "risk_assessment": {"risk_level": str, "risk_factors": [str, ...], "flag_count": int}
# }
```

See `README.md` for the full input specification (types, valid ranges, descriptions).  
See `app.py` for a working reference implementation of a complete UI.

---

## Coding Conventions

- **Python 3.10+** with type hints on all public functions
- **No emojis** anywhere in code or output
- **`random_state=42`** for all stochastic operations
- **Recall** is the primary metric — never optimise for accuracy alone
- Keep `bridge.py` clean: it must be importable without side effects
- No print statements in library code — use `logging.getLogger(__name__)`
- All plots saved to `plots/`, all artifacts saved to `models/`
- Run `python verify_setup.py` before opening a pull request

---

## Key Design Decisions

See `PROJECT_LOG.md → Section 7` for the full rationale. Short version:

| Decision | Why |
|----------|-----|
| Recall over Accuracy | False negatives (missed disease) are more dangerous than false positives (unnecessary follow-up) |
| F2-score for threshold tuning | Weights Recall 2× more than Precision, matching the clinical cost asymmetry |
| Separate 15% validation split | Threshold tuned on val set → test-set metrics are unbiased |
| `class_weight='balanced'` | Biases models toward detecting positive cases without oversampling |
| Three models compared | Ensures the best-performing model is always selected, not assumed |

---

## Questions

Open an issue on GitHub or message Hassan directly.
