# MediAssist - AI-Powered Medical Diagnosis Support System

Backend engine for cardiovascular disease prediction using a **hybrid approach**:
a rule-based clinical knowledge engine combined with three optimised machine learning
models (Logistic Regression, Random Forest, LightGBM). The system is optimised for
**Recall (Sensitivity)**, the medically correct primary metric for disease screening.

---

## Why Recall, Not Accuracy?

**Accuracy** treats every prediction error as equal. In cardiovascular screening,
errors are not equal:

| Error Type | What it means | Clinical consequence |
|------------|---------------|----------------------|
| **False Negative** (missed disease) | Model says "healthy"; patient has disease | Patient goes untreated, potentially fatal |
| **False Positive** (false alarm) | Model says "disease"; patient is healthy | Patient receives a follow-up consultation, inconvenient but not dangerous |

**Recall (Sensitivity) = TP / (TP + FN)** measures the proportion of actual disease
cases the model correctly identifies. Maximising Recall minimises missed diagnoses.
We accept more false positives as the deliberate trade-off for not missing sick patients.

**F2-score** (used for threshold tuning) weights Recall twice as heavily as Precision
(beta=2), reflecting this same cost asymmetry in a single optimisation objective.

---

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

---

## Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.\venv\Scripts\activate.bat`
- **macOS/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Download the Cardiovascular Disease dataset from
[Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease)
and place `cardio_train.csv` in the `data/` directory:

```
MediAssist/
  data/
    cardio_train.csv
```

Alternatively, if you have `kagglehub` installed and a Kaggle API token
configured, `main.py` will download the dataset automatically.

### 4. Verify your setup

```bash
python verify_setup.py
```

This checks Python version, all packages, the dataset, model artifacts, and an end-to-end `bridge.py` call. Fix any failures before continuing.

### 5. Run the training pipeline (optional)

> **The trained model artifacts are already included in the repository** (`models/*.pkl`).
> You only need to run this if you want to retrain from scratch or the dataset has changed.
> Skip to Step 6 if you just want to use the pre-bundled model.

```bash
python main.py
```

This will:
1. Check for / download the dataset
2. Preprocess the data (outlier removal, age conversion, BMI, feature engineering, normalisation)
3. Train Logistic Regression, Random Forest (tuned), and LightGBM (tuned)
4. Evaluate all three models and select the one with the highest Recall
5. Tune the classification threshold using F2-score on a **held-out validation set**
6. Generate five evaluation plots in `plots/`
7. Export the trained model, scaler, feature names, and threshold to `models/`
8. Write `PROJECT_LOG.md` with full technical metrics and methodology justification

### 6. Launch the demo interface (optional)

```bash
streamlit run app.py
```

Opens a browser-based form for testing the prediction API manually.
This is a temporary reference implementation; see [Frontend Integration](#using-bridgepy-frontend-integration)
for how to build the production frontend.

---

## Project Structure

```
MediAssist/
├── data/
│   └── cardio_train.csv              # Input dataset (not tracked in git)
├── models/
│   ├── final_model.pkl               # Trained best model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── feature_names.pkl             # Ordered feature name list (14 features)
│   └── threshold.pkl                 # F2-optimised classification threshold
├── plots/
│   ├── learning_curves.png           # Recall vs. training set size
│   ├── confusion_matrix.png          # Confusion matrix at tuned threshold
│   ├── feature_importance.png        # Feature importance ranked bar chart
│   ├── precision_recall_curve.png    # PR curve with default + optimal threshold
│   └── roc_curve.png                 # ROC curve for all three models
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Loading, cleaning, feature engineering, splits
│   ├── knowledge_engine.py           # Rule-based clinical risk assessment
│   ├── ml_models.py                  # Training, tuning, selection, threshold search
│   └── evaluation.py                 # Plot generation (5 plots)
├── app.py                            # Temporary Streamlit demo interface
├── bridge.py                         # Frontend integration interface
├── config.py                         # Central configuration (paths, hyperparameters)
├── main.py                           # Training pipeline orchestrator
├── verify_setup.py                   # Environment health check script
├── requirements.txt                  # Python dependencies
├── CONTRIBUTING.md                   # Partner onboarding and conventions guide
├── PROJECT_LOG.md                    # Auto-generated technical report
└── README.md                         # This file
```

---

## Using bridge.py (Frontend Integration)

`bridge.py` is the **only file a frontend application needs to import**.
Model artifacts are pre-bundled in `models/` so you can call `get_prediction()` immediately after cloning.
Artifacts are lazy-loaded on the first call and cached for all subsequent calls.
No need to manage model state yourself.

### Function Signature

```python
from bridge import get_prediction

result = get_prediction(patient_data: dict) -> dict
```

### Input Format

`patient_data` must be a dictionary with the following 11 keys:

| Key           | Type  | Range   | Description                                    |
|---------------|-------|---------|------------------------------------------------|
| `age`         | int   | 1–120   | Patient age in years                           |
| `gender`      | int   | 1–2     | 1 = female, 2 = male                          |
| `height`      | int   | 100–220 | Height in centimetres                          |
| `weight`      | float | 30–200  | Weight in kilograms                            |
| `ap_hi`       | int   | 60–250  | Systolic blood pressure (mmHg)                 |
| `ap_lo`       | int   | 40–160  | Diastolic blood pressure (mmHg)                |
| `cholesterol` | int   | 1–3     | 1 = normal, 2 = above normal, 3 = well above  |
| `gluc`        | int   | 1–3     | 1 = normal, 2 = above normal, 3 = well above  |
| `smoke`       | int   | 0–1     | 0 = non-smoker, 1 = smoker                    |
| `alco`        | int   | 0–1     | 0 = no alcohol, 1 = alcohol intake            |
| `active`      | int   | 0–1     | 0 = inactive, 1 = physically active           |

> **Note:** BMI, pulse pressure (`ap_hi − ap_lo`), and an age×BMI interaction
> term are computed internally by `bridge.py`. You do not supply them.

### Output Format

```python
{
    "prediction": 1,              # 0 = no disease  |  1 = cardiovascular disease detected
    "probability": 0.7958,        # Model probability of disease (0.0–1.0)
    "threshold_used": 0.1371,     # F2-optimised threshold (loaded from models/threshold.pkl)
    "risk_assessment": {
        "risk_level": "High Risk",    # "High Risk"  |  "Moderate Risk"  |  "Low Risk"
        "risk_factors": [
            "Hypertension detected (systolic: 150, diastolic: 95)",
            "Obesity indicated (BMI: 31.8)",
            "Cholesterol well above normal",
            "Elevated lifestyle risk (age 58, sedentary)"
        ],
        "flag_count": 4               # Number of clinical rules triggered (rule-based engine)
    }
}
```

> **`prediction` vs `probability`:** `prediction` already applies the tuned threshold.
> Use `probability` for a risk percentage bar or dial in your UI; use `prediction`
> for the binary yes/no disease label.

### Example Call

```python
from bridge import get_prediction

patient = {
    "age": 58,
    "gender": 2,
    "height": 170,
    "weight": 92,
    "ap_hi": 150,
    "ap_lo": 95,
    "cholesterol": 3,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 0,
}

result = get_prediction(patient)
print(result["prediction"])        # 1  (disease detected)
print(result["probability"])       # 0.7958
print(result["threshold_used"])    # 0.1371
print(result["risk_assessment"])   # {"risk_level": "High Risk", ...}
```

### Error Handling

`get_prediction()` raises:
- `ValueError`: required keys are missing, or a value is out of its defined range
- `TypeError`: a value is not numeric
- `FileNotFoundError`: model artifacts not found; run `python main.py` first

---

## Methodology Summary

| Design choice | Justification |
|---------------|---------------|
| **Recall as primary metric** | Minimises false negatives (missed diagnoses), the highest-cost error in medical screening. Accuracy is misleading because it weights all errors equally. |
| **F2-score for threshold tuning** | Weights Recall 2× more than Precision, reflecting the asymmetric cost of false negatives over false positives. |
| **Separate 15% validation split** | Threshold tuned on val set; test-set metrics are never contaminated and reflect true generalisation performance. |
| **70/15/15 train/val/test split** | Standard practice for systems with a post-training search step (threshold tuning). |
| **Three models compared** | LR (interpretable baseline), RF (ensemble), LightGBM (gradient boosting, typically strongest on tabular data). |
| **RandomizedSearchCV (30 iter, 5-fold)** | Scored on Recall; avoids exhaustive grid-search cost while still exploring the hyperparameter space. |
| **`class_weight='balanced'`** | Counters class imbalance without oversampling; biases all models toward detecting positive cases. |
| **StandardScaler on train only** | Prevents data leakage; val and test transforms use training-set statistics only. |
| **`feature_names.pkl`** | Ensures feature order at inference exactly matches training order. |

---

## Notes

- All stochastic operations use `random_state=42` for reproducibility.
- Full classification reports, confusion matrices, and ethical disclaimers are in `PROJECT_LOG.md`.
- This is an educational decision-support tool. It is **not** intended for clinical use.
- See `app.py` for a working reference implementation of a UI built on top of `bridge.py`.

## Model artifacts and Git LFS

The trained model artifact `models/final_model.pkl` is large and is tracked using
Git Large File Storage (Git LFS). If you clone this repository and want to fetch
the pre-trained model artifact, run:

```bash
git clone https://github.com/Ramendan/MediAssist.git
cd MediAssist
git lfs install
git lfs pull
```

If you cannot or do not want to fetch LFS objects, you can retrain the model
locally by running `python main.py` to generate the artifacts under `models/`.
Note that training may take 10–30 minutes depending on your machine.
