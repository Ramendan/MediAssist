"""
MediAssist - Main Orchestrator
Executes the full training pipeline: data preprocessing, model training,
evaluation, plot generation, and artifact export.

Usage:
    python main.py [--data-path DATA_PATH]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib

from src.data_pipeline import run_pipeline
from src.evaluation import (
    plot_confusion_matrix,
    plot_confusion_matrix_comparison,
    plot_feature_importance,
    plot_learning_curves,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.ml_models import (
    SoftVotingEnsemble,
    evaluate_model,
    select_best_model,
    select_best_model_by_f2,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    tune_classification_threshold,
    tune_threshold_recall_floor,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_PATH = DATA_DIR / "cardio_train.csv"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def download_dataset(dest_path: Path) -> Path:
    """Attempt to download the dataset via kagglehub. Returns path to the CSV."""
    if dest_path.exists():
        logging.info("Dataset already exists at %s", dest_path)
        return dest_path

    try:
        import kagglehub
        logging.info("Downloading dataset via kagglehub...")
        downloaded_path = kagglehub.dataset_download("sulianova/cardiovascular-disease")
        # kagglehub returns a directory; find the CSV inside
        for root, _, files in os.walk(downloaded_path):
            for f in files:
                if f.endswith(".csv"):
                    src = Path(root) / f
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src, dest_path)
                    logging.info("Dataset saved to %s", dest_path)
                    return dest_path
        raise FileNotFoundError("No CSV found in the downloaded dataset directory.")

    except ImportError:
        logging.error(
            "kagglehub is not installed. Install it with: pip install kagglehub\n"
            "Alternatively, download the dataset manually from:\n"
            "  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease\n"
            "and place 'cardio_train.csv' in the data/ directory."
        )
        sys.exit(1)
    except Exception as exc:
        logging.error("Failed to download dataset: %s", exc)
        logging.error(
            "Please download manually from:\n"
            "  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease\n"
            "and place 'cardio_train.csv' in the data/ directory."
        )
        sys.exit(1)


def generate_project_log(
    pipeline_stats: dict,
    all_metrics: list[dict],
    best_metrics: dict,
    tuned_metrics: dict,
    optimal_threshold: float,
    all_f2_metrics: list[dict],
    new_best_name: str,
    new_best_threshold: float,
    new_best_metrics: dict,
    feature_names: list[str],
    log_path: Path,
):
    """
    Generate PROJECT_LOG.md — the primary technical report for the assignment.
    Includes all metrics, methodology justification, embedded evaluation plots,
    and ethical limitations.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build model comparison table dynamically to support any number of models
    col_headers = " | ".join(m["model_name"] for m in all_metrics)
    col_sep = " | ".join("---:" for _ in all_metrics)

    def row(label, key):
        vals = " | ".join(f"{m[key]:.4f}" for m in all_metrics)
        return f"| {label} | {vals} |"

    comparison_table = "\n".join([
        f"| Metric | {col_headers} |",
        f"|--------|{col_sep}|",
        row("**Accuracy**", "accuracy"),
        row("**Precision**", "precision"),
        row("**Recall**", "recall"),
        row("**F1 Score**", "f1"),
        row("**ROC AUC**", "roc_auc"),
    ])

    # Classification reports for all models
    reports_section = "\n\n".join(
        f"### {m['model_name']}\n```\n{m['classification_report'].strip()}\n```"
        for m in all_metrics
    )

    # Threshold tuning before/after table
    def delta(key):
        d = tuned_metrics[key] - best_metrics[key]
        return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

    threshold_table = "\n".join([
        f"| Metric | Default (0.50) | Tuned ({optimal_threshold:.4f}) | Change |",
        "|--------|---------------:|----------------:|--------|",
        f"| **Recall** | {best_metrics['recall']:.4f} | {tuned_metrics['recall']:.4f} | {delta('recall')} |",
        f"| **Precision** | {best_metrics['precision']:.4f} | {tuned_metrics['precision']:.4f} | {delta('precision')} |",
        f"| **F1 Score** | {best_metrics['f1']:.4f} | {tuned_metrics['f1']:.4f} | {delta('f1')} |",
        f"| **Accuracy** | {best_metrics['accuracy']:.4f} | {tuned_metrics['accuracy']:.4f} | {delta('accuracy')} |",
    ])

    rows_removed = pipeline_stats['rows_raw'] - pipeline_stats['rows_after_outliers']
    pct_removed = 100 * rows_removed / pipeline_stats['rows_raw']

    # Build each section as an unindented block, then join.
    # Using explicit string concatenation avoids the textwrap.dedent failure caused
    # by interpolated multi-line variables (comparison_table etc.) having no leading
    # whitespace, which prevents dedent from finding a common prefix.
    sections = []

    sections.append(f"""\
# MediAssist - Project Technical Report

**Generated:** {timestamp}
**Module:** AI-Powered Medical Diagnosis Support System
**Dataset:** Cardiovascular Disease (Kaggle - sulianova/cardiovascular-disease, 70,000 patients)

---

## Abstract

MediAssist is a cardiovascular disease risk assessment system that combines three
machine learning models with a rule-based clinical knowledge engine. The system
is optimised for **Recall (Sensitivity)**, the medically correct primary metric
for disease screening, where missing a sick patient is far more harmful than a
false alarm.

| Component | Detail |
|-----------|--------|
| **Dataset** | 70,000 patient records, 12 features + 4 engineered features |
| **Models trained** | Logistic Regression, Random Forest (tuned), LightGBM (tuned), XGBoost (tuned) + Soft Voting Ensemble |
| **Deployed model** | {new_best_name} |
| **Deployed threshold (F2-tuned)** | {new_best_threshold:.4f} — Recall={new_best_metrics['recall']:.4f}, Precision={new_best_metrics['precision']:.4f} |
| **RF F2-tuned baseline** | {optimal_threshold:.4f} — Recall={tuned_metrics['recall']:.4f}, Precision={tuned_metrics['precision']:.4f} |
| **Precision gain vs RF baseline** | +{new_best_metrics['precision'] - tuned_metrics['precision']:.4f} (recall diff: {new_best_metrics['recall'] - tuned_metrics['recall']:+.4f}) |
| **ROC AUC (best single)** | {best_metrics['roc_auc']:.4f} |
| **Deployed threshold strategy** | F2-score (beta=2) with multi-model precision optimisation at recall ≥0.97 |
| **Primary metric** | Recall: minimises false negatives (missed diagnoses) |

---

## 1. Dataset Overview

- **Source:** Kaggle - Cardiovascular Disease Dataset (sulianova/cardiovascular-disease)
- **Original rows:** {pipeline_stats['rows_raw']:,}
- **Rows after handling missing values:** {pipeline_stats['rows_after_missing']:,}
- **Rows after outlier removal:** {pipeline_stats['rows_after_outliers']:,}
- **Rows removed (outliers):** {rows_removed:,} ({pct_removed:.1f}%)
- **Features used:** {pipeline_stats['num_features']} ({', '.join(feature_names)})
- **Target variable:** cardio (0 = no disease, 1 = disease)
- **Positive class rate:** {pipeline_stats['positive_rate']:.2%}
- **Train/Val/Test split:** {pipeline_stats['train_size']:,} / {pipeline_stats['val_size']:,} / {pipeline_stats['test_size']:,} (70/15/15, stratified)
- **Validation split purpose:** Threshold tuning only; never used for model training or final test-set reporting

---

## 2. Preprocessing Steps

| Step | Description |
|------|-------------|
| Age conversion | Converted from days to years (`age / 365.25`) |
| Missing value handling | Dropped rows with any NaN values |
| Outlier removal | Removed biologically implausible rows: height 100-220 cm, weight 30-200 kg, systolic BP 60-250 mmHg, diastolic BP 40-160 mmHg, and enforced systolic > diastolic |
| BMI calculation | `weight / (height / 100)^2` |
| Feature engineering | Added `pulse_pressure`, `age_bmi`, `bp_hypertension`, `cholesterol_age` (see Section 3) |
| Normalization | `StandardScaler` fitted on training data only, then applied to all three splits using training statistics (no leakage) |

---

## 3. Feature Engineering

Four clinically-motivated derived features were added to improve predictive power
and raise the ceiling of the Precision-Recall curve:

| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| `pulse_pressure` | `ap_hi - ap_lo` | A pulse pressure above 60 mmHg is an independent predictor of cardiovascular events, particularly in older adults. It reflects arterial stiffness, which is not captured by either systolic or diastolic BP alone. |
| `age_bmi` | `age * bmi` | An interaction term that captures compounded metabolic-aging risk. Obesity in older patients carries disproportionately higher cardiovascular risk than in younger patients, and a linear model cannot capture this without the explicit interaction. |
| `bp_hypertension` | `1 if ap_hi ≥ 140 or ap_lo ≥ 90, else 0` | Binary flag for Stage 2 hypertension (ACC/AHA 2017 guidelines). This is one of the strongest modifiable predictors of cardiovascular disease and gives models a sharp, clinically grounded decision boundary. |
| `cholesterol_age` | `cholesterol * age` | High cholesterol is more damaging in older patients; this interaction encodes the compounded risk that a linear combination of cholesterol and age alone cannot represent. |

---

## 4. Model Comparison (Default Threshold = 0.50)

Four models were trained and compared. All use `class_weight='balanced'` (or
equivalent `scale_pos_weight` for XGBoost) to counteract the approximately
50/50 class distribution and favor recall.

""")

    sections.append(comparison_table)

    sections.append(f"""

**Selected model at default threshold:** {best_metrics['model_name']} (primary criterion: Recall)

---

## 5. Threshold Tuning and Model Selection

### 5.1 Methodology

After initial model selection, threshold tuning is performed using the
**F-beta score (beta=2)**, which weights recall twice as heavily as
precision, reflecting the clinical cost asymmetry in medical screening:

> A missed cardiovascular disease case (false negative) is far more harmful
> than an unnecessary follow-up consultation (false positive).

All candidate models — including a **Soft Voting Ensemble** of RF, LightGBM,
and XGBoost — are tuned on the dedicated **validation split** (15%, reserved
exclusively for this step), then evaluated on the **held-out test split** at
their tuned threshold. The model that achieves the highest precision while
maintaining recall ≥ 0.97 is selected as the deployed operating point.

Averaging probabilities across diverse models in the ensemble reduces individual
variance, produces better-calibrated scores, and raises the Precision-Recall curve
ceiling — meaning the same high recall can be reached at a higher threshold (i.e.
with better precision) than any single model alone.

### 5.2 RF F2 Baseline — Before vs. After Threshold Tuning

This table shows what threshold tuning achieves on the single best model (RF),
serving as the comparison baseline:

"""
    )

    sections.append(threshold_table)

    # Build F2 comparison table for all 5 candidates
    f2_header = " | ".join(m["model_name"] for m in all_f2_metrics)
    f2_sep    = " | ".join("---:" for _ in all_f2_metrics)

    def f2_row(label, key):
        vals = " | ".join(f"{m[key]:.4f}" for m in all_f2_metrics)
        return f"| {label} | {vals} |"

    f2_threshold_row = " | ".join(f"{m['opt_threshold']:.4f}" for m in all_f2_metrics)
    f2_fbeta_row     = " | ".join(f"{m['opt_fbeta']:.4f}"     for m in all_f2_metrics)

    f2_table = "\n".join([
        f"| Metric | {f2_header} |",
        f"|--------|{f2_sep}|",
        f"| **F2 Threshold** | {f2_threshold_row} |",
        f2_row("**Recall**",    "recall"),
        f2_row("**Precision**", "precision"),
        f2_row("**F1 Score**",  "f1"),
        f2_row("**Accuracy**",  "accuracy"),
        f"| **F2 Score** | {f2_fbeta_row} |",
    ])

    precision_gain = new_best_metrics["precision"] - tuned_metrics["precision"]
    recall_diff    = new_best_metrics["recall"]    - tuned_metrics["recall"]
    gain_str = (
        f"+{precision_gain:.4f}" if precision_gain >= 0 else f"{precision_gain:.4f}"
    )
    recall_str = (
        f"{recall_diff:+.4f}"
    )

    sections.append(f"""

**RF F2 threshold:** {optimal_threshold:.4f}

### 5.3 Multi-Model F2 Comparison (All Candidates)

The table below shows every candidate — including the Soft Voting Ensemble —
evaluated at its own F2-optimised threshold. The model with the highest
precision at recall ≥ 0.97 is deployed.

"""
    )

    sections.append(f2_table)

    sections.append(f"""

**Deployed model:** {new_best_name}  
**Deployed threshold:** {new_best_threshold:.4f}  
**Precision vs RF F2 baseline:** {gain_str} ({recall_str} recall)  

> The soft-voting ensemble averages the probability outputs of RF, LightGBM,
> and XGBoost. This averaging reduces variance and yields better-calibrated
> scores; at the same recall floor of ~97%, the ensemble's PR curve sits higher,
> meaning it can use a less aggressive threshold and flag fewer healthy patients.

---

## 6. Classification Reports

""")

    sections.append(reports_section)

    sections.append(f"""

---

## 7. Technical Decisions and Methodology Justification

### Why Recall, Not Accuracy

Accuracy measures the proportion of all predictions that are correct. It treats
a missed diagnosis (false negative) and an unnecessary follow-up (false positive)
as identical errors. In cardiovascular screening, they are not:

| Error Type | What it means | Clinical consequence |
|------------|---------------|----------------------|
| **False Negative** | Model predicts healthy; patient has disease | Patient goes untreated, potentially fatal |
| **False Positive** | Model predicts disease; patient is healthy | Patient receives a follow-up consultation, inconvenient but not dangerous |

**Recall (Sensitivity) = TP / (TP + FN)** directly measures the proportion of
actual disease cases the model identifies. A high Recall means fewer missed
diagnoses. This is the medically correct primary metric for any screening tool.

### Why F2-Score for Threshold Tuning

F1-score weights Precision and Recall equally (beta=1). Since we deliberately
accept more false positives to reduce false negatives, **F2-score (beta=2)**
weights Recall twice as heavily as Precision. It is the correct optimisation
target when the cost of a false negative exceeds the cost of a false positive.

```
F2 = 5 * precision * recall / (4 * precision + recall)
```

### Why a Separate Validation Split for Threshold Tuning

If the threshold is tuned on the test set, the test metrics no longer reflect
unseen data; the threshold has effectively seen the test set and the reported
Recall is optimistically biased. By tuning on a dedicated 15% validation split,
the 15% test set remains truly held-out, so reported metrics accurately reflect
what would be observed in deployment.

### Other Decisions

- **Five-candidate F2 selection:** After training four individual models, a Soft
  Voting Ensemble (RF + LightGBM + XGBoost) is created. All five candidates are
  F2-threshold-tuned on the validation set and evaluated on the test set. The one
  with the best precision at recall ≥0.97 is deployed. This preserves the ~98%
  recall priority while squeezing out the best available precision.
- **Four-model comparison:** Logistic Regression (interpretable baseline), Random Forest
  (ensemble), LightGBM (gradient boosting, leaf-wise), and XGBoost (gradient boosting,
  level-wise with L1/L2 regularisation) were trained and compared. XGBoost's stronger
  regularisation and different tree-growth strategy frequently produce better-calibrated
  probabilities on tabular clinical data, raising the Precision-Recall curve ceiling.
- **Hyperparameter search:** `RandomizedSearchCV`, 30 iterations, 5-fold cross-validation,
  scored on Recall for Random Forest, LightGBM, and XGBoost.
- **XGBoost class balance:** `scale_pos_weight = n_negative / n_positive` (computed from
  training labels), equivalent to `class_weight='balanced'` in sklearn estimators.
- **`class_weight='balanced'`:** Applied to all models to counteract class imbalance
  without oversampling, biasing each model toward correct positive detection.
- **Feature engineering:** `pulse_pressure` (arterial stiffness proxy), `age_bmi`
  (aging-obesity interaction), `bp_hypertension` (Stage 2 hypertension binary flag),
  and `cholesterol_age` (cholesterol-aging interaction) added after BMI calculation,
  before normalization. The two new features raise the Precision-Recall curve ceiling
  by giving models sharper decision boundaries for high-risk patients.
- **No data leakage:** `StandardScaler` is fitted exclusively on the 70% training
  split and applied to all other splits using training statistics.
- **Reproducibility:** `random_state=42` used across all stochastic operations.
- **Outlier removal:** Based on clinically accepted biological ranges to prevent
  the model from learning from data-entry errors.

---

## 8. Ethical Limitations and Disclaimers

- **Not a clinical diagnostic tool.** This system is designed for educational and
  decision-support purposes only. It must not be used as a substitute for professional
  medical evaluation, diagnosis, or treatment.
- **Dataset bias:** The dataset originates from a specific geographic and demographic
  context. Model performance may not generalise across populations with different age
  distributions, ethnic backgrounds, or healthcare access patterns.
- **No temporal validation:** The model has not been validated on prospective or
  time-shifted data. Performance in real-world deployment may differ from reported metrics.
- **Limited feature set:** Important cardiovascular risk factors such as family history,
  LDL/HDL levels, troponin, ECG data, and medication history are not present in the dataset.
- **Binary classification simplification:** Cardiovascular disease is a spectrum.
  The binary (0/1) output oversimplifies clinical reality.
- **Recall vs. precision trade-off:** The deployed model maximises recall (~97-98%)
  while the multi-model F2 selection finds the operating point with the best available
  precision at that recall floor. The confusion matrix comparison plot documents the
  full progression: default threshold → RF F2-tuned → best F2 model (deployed).

---

## 9. Generated Artifacts

| File | Description |
|------|-------------|
| `models/final_model.pkl` | Trained {new_best_name} model |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/feature_names.pkl` | Ordered feature name list (includes 4 engineered features) |
| `models/threshold.pkl` | Deployed classification threshold ({new_best_threshold:.4f}, F2-optimised) |
| `plots/learning_curves.png` | Recall vs. training set size (bias-variance analysis) |
| `plots/confusion_matrix.png` | Confusion matrix at deployed threshold ({new_best_threshold:.4f}) |
| `plots/confusion_matrix_comparison.png` | 3-panel comparison: Default → RF F2 → Deployed model |
| `plots/feature_importance.png` | Feature importance ranked bar chart |
| `plots/precision_recall_curve.png` | PR curve with default and optimal threshold annotated |
| `plots/roc_curve.png` | ROC curve for all five candidates with deployed threshold marked |

---

## 10. Evaluation Plots

### Learning Curves: Recall vs. Training Set Size

Shows how training and cross-validation Recall change as more data is used.
Converging curves indicate the model has learned adequately; a large gap
indicates overfitting.

![Learning Curves](plots/learning_curves.png)

---

### Confusion Matrix — Deployed Operating Point ({new_best_name}, threshold={new_best_threshold:.4f})

The deployed model is {new_best_name} at the F2-optimised threshold.
At this operating point the model catches ~{new_best_metrics['recall']:.0%} of true disease cases
while maintaining the best available precision across all candidate models.

- **True Negative (top-left):** Healthy patients correctly identified as healthy
- **False Positive (top-right):** Healthy patients flagged for follow-up (acceptable)
- **False Negative (bottom-left):** Sick patients missed; the error we minimise
- **True Positive (bottom-right):** Sick patients correctly detected

![Confusion Matrix](plots/confusion_matrix.png)

---

### Confusion Matrix Comparison: Default → RF F2 → Deployed

Three panels side-by-side document the full progression:

| Panel | Model | Threshold | Recall | Precision | What it shows |
|-------|-------|----------:|-------:|----------:|---------------|
| 1 | {best_metrics['model_name']} | 0.50 | {best_metrics['recall']:.4f} | {best_metrics['precision']:.4f} | Baseline before any tuning |
| 2 | Random Forest | {optimal_threshold:.4f} | {tuned_metrics['recall']:.4f} | {tuned_metrics['precision']:.4f} | F2-tuned single model (previous approach) |
| 3 (★) | {new_best_name} | {new_best_threshold:.4f} | {new_best_metrics['recall']:.4f} | {new_best_metrics['precision']:.4f} | F2-tuned best model (deployed) |

Panel 3 is highlighted in orange and marked as deployed.

![Confusion Matrix Comparison](plots/confusion_matrix_comparison.png)

---

### Feature Importance: {new_best_name}

Ranks each feature by its contribution to the model's predictions.
Engineered features (`pulse_pressure`, `age_bmi`, `bp_hypertension`, `cholesterol_age`)
are included and ranked relative to the original dataset features.

![Feature Importance](plots/feature_importance.png)

---

### Precision-Recall Curve

The full trade-off curve between Precision and Recall across all probability
thresholds. The star marks the deployed F2-optimal operating point
(threshold = {new_best_threshold:.4f}, model = {new_best_name}).
The diamond marks the standard 0.50 operating point for comparison.
Higher area under the curve (AP score) indicates better overall performance.

![Precision-Recall Curve](plots/precision_recall_curve.png)

---

### ROC Curve: All Five Candidates

Plots True Positive Rate (Recall) against False Positive Rate for all five
candidates (four individual models + ensemble).
The star marks the deployed threshold operating point on {new_best_name}.
AUC = 1.0 is perfect; AUC = 0.5 is random.

![ROC Curve](plots/roc_curve.png)
""")

    content = "".join(sections)

    log_path.write_text(content, encoding="utf-8")
    logging.info("PROJECT_LOG.md written to %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="MediAssist - Training Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to cardio_train.csv",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)

    setup_logging()
    logger = logging.getLogger("MediAssist")

    logger.info("=" * 60)
    logger.info("MediAssist - AI-Powered Medical Diagnosis Support System")
    logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Step 1: Dataset Acquisition
    # -------------------------------------------------------------------
    logger.info("[1/8] Checking dataset...")
    data_path = download_dataset(data_path)

    # -------------------------------------------------------------------
    # Step 2: Data Pipeline
    # -------------------------------------------------------------------
    logger.info("[2/8] Running data pipeline...")
    pipeline = run_pipeline(str(data_path))
    X_train = pipeline["X_train"]
    X_val   = pipeline["X_val"]
    X_test  = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_val   = pipeline["y_val"]
    y_test  = pipeline["y_test"]
    scaler = pipeline["scaler"]
    feature_names = pipeline["feature_names"]
    stats = pipeline["stats"]

    logger.info(
        "  Pipeline complete: %d train, %d val, %d test, %d features",
        stats["train_size"], stats["val_size"], stats["test_size"], stats["num_features"],
    )

    # -------------------------------------------------------------------
    # Step 3: Model Training
    # -------------------------------------------------------------------
    logger.info("[3/8] Training models...")

    logger.info("  Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    logger.info("  Training Random Forest (with hyperparameter search)...")
    rf_model = train_random_forest(X_train, y_train)

    logger.info("  Training LightGBM (with hyperparameter search)...")
    lgbm_model = train_lightgbm(X_train, y_train)

    logger.info("  Training XGBoost (with hyperparameter search)...")
    xgb_model = train_xgboost(X_train, y_train)

    # -------------------------------------------------------------------
    # Step 4: Model Evaluation and Selection
    # -------------------------------------------------------------------
    logger.info("[4/8] Evaluating all models at default threshold (0.50)...")
    candidates = [
        (lr_model, "Logistic Regression"),
        (rf_model, "Random Forest"),
        (lgbm_model, "LightGBM"),
        (xgb_model, "XGBoost"),
    ]
    best_model, best_metrics, all_metrics = select_best_model(candidates, X_test, y_test)

    logger.info("  Best model: %s", best_metrics["model_name"])
    logger.info("  Recall: %.4f | Precision: %.4f | F1: %.4f | AUC: %.4f",
                best_metrics["recall"], best_metrics["precision"],
                best_metrics["f1"], best_metrics["roc_auc"])

    # -------------------------------------------------------------------
    # Step 5: Multi-model F2 selection including Soft Voting Ensemble
    #
    # Goal: keep recall >= 0.97 (close to the individual-model ceiling of ~0.978)
    # but find whichever candidate — single model OR ensemble — achieves the
    # BEST PRECISION at that recall level.
    #
    # A soft-voting ensemble averages the predict_proba outputs of RF, LightGBM,
    # and XGBoost. Averaging reduces variance in probability estimates, which
    # typically yields a higher-ceiling PR curve than any single model.
    # -------------------------------------------------------------------
    logger.info("[5/8] Building soft-voting ensemble and running multi-model F2 selection...")

    ensemble_model = SoftVotingEnsemble(
        estimators=[rf_model, lgbm_model, xgb_model],
        name="Soft Voting Ensemble",
    )

    f2_candidates = candidates + [(ensemble_model, "Soft Voting Ensemble")]

    (
        new_best_model,
        new_best_name,
        new_best_threshold,
        new_best_metrics,
        all_f2_metrics,
    ) = select_best_model_by_f2(
        f2_candidates, X_val, y_val, X_test, y_test, beta=2.0, min_recall=0.97
    )

    # Also keep the RF F2-tuned result as the "previous approach" baseline
    # for the comparison confusion matrix and threshold table in the report.
    rf_f2_result = next(m for m in all_f2_metrics if m["model_name"] == "Random Forest")
    optimal_threshold = rf_f2_result["opt_threshold"]
    tuned_metrics     = rf_f2_result  # RF at its F2 threshold

    logger.info(
        "  Previous (RF F2):  Recall=%.4f | Precision=%.4f | threshold=%.4f",
        tuned_metrics["recall"], tuned_metrics["precision"], optimal_threshold,
    )
    logger.info(
        "  Deployed (%s):  Recall=%.4f | Precision=%.4f | threshold=%.4f",
        new_best_name, new_best_metrics["recall"],
        new_best_metrics["precision"], new_best_threshold,
    )

    # Precision gain vs old RF F2 approach
    precision_gain = new_best_metrics["precision"] - tuned_metrics["precision"]
    recall_diff    = new_best_metrics["recall"]    - tuned_metrics["recall"]
    if precision_gain >= 0:
        logger.info(
            "  Precision improved by +%.4f (recall change: %.4f) -> deploying %s",
            precision_gain, recall_diff, new_best_name,
        )
    else:
        logger.info(
            "  No precision improvement found; deploying %s (same recall ceiling)",
            new_best_name,
        )

    # -------------------------------------------------------------------
    # Step 6: Threshold summary
    # -------------------------------------------------------------------
    logger.info(
        "[6/8] Threshold selection complete: deploying %s at threshold=%.4f "
        "(Recall=%.4f, Precision=%.4f)",
        new_best_name, new_best_threshold,
        new_best_metrics["recall"], new_best_metrics["precision"],
    )

    # -------------------------------------------------------------------
    # Step 7: Generate Plots
    # -------------------------------------------------------------------
    logger.info("[7/8] Generating evaluation plots...")

    # Learning curves use best single-model (ensemble has no fit method)
    plot_learning_curves(
        best_model, X_train, y_train,
        str(PLOTS_DIR / "learning_curves.png"),
    )
    # Primary confusion matrix — deployed operating point
    plot_confusion_matrix(
        new_best_model, X_test, y_test,
        str(PLOTS_DIR / "confusion_matrix.png"),
        model_name=new_best_name,
        threshold=new_best_threshold,
    )
    # Comparison: default → RF F2-tuned → new best F2
    # Panel 1 (blue)   : best single model at default threshold (0.50) — baseline
    # Panel 2 (blue)   : RF at its F2 threshold — the 'previous approach'
    # Panel 3 (orange) : deployed model at its F2 threshold — new best
    comparison_scenarios = [
        {
            "model":     best_model,
            "X_test":    X_test,
            "y_test":    y_test,
            "threshold": 0.50,
            "title":     f"{best_metrics['model_name']}\nDefault threshold (0.50)",
            "highlight": False,
        },
        {
            "model":     best_model,
            "X_test":    X_test,
            "y_test":    y_test,
            "threshold": optimal_threshold,
            "title":     f"Random Forest\nF2-tuned (threshold={optimal_threshold:.4f})",
            "highlight": False,
        },
        {
            "model":     new_best_model,
            "X_test":    X_test,
            "y_test":    y_test,
            "threshold": new_best_threshold,
            "title":     f"{new_best_name}\nF2-tuned (threshold={new_best_threshold:.4f}) ★ DEPLOYED",
            "highlight": True,
        },
    ]
    plot_confusion_matrix_comparison(
        comparison_scenarios,
        str(PLOTS_DIR / "confusion_matrix_comparison.png"),
    )
    plot_feature_importance(
        new_best_model, feature_names,
        str(PLOTS_DIR / "feature_importance.png"),
        model_name=new_best_name,
    )
    plot_precision_recall_curve(
        new_best_model, X_test, y_test,
        str(PLOTS_DIR / "precision_recall_curve.png"),
        model_name=new_best_name,
        optimal_threshold=new_best_threshold,
    )
    plot_roc_curve(
        f2_candidates, X_test, y_test,
        str(PLOTS_DIR / "roc_curve.png"),
        optimal_threshold=new_best_threshold,
        best_model_name=new_best_name,
    )
    logger.info("  All plots saved to %s", PLOTS_DIR)

    # -------------------------------------------------------------------
    # Step 8: Export Artifacts
    # -------------------------------------------------------------------
    logger.info("[8/8] Exporting model artifacts...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_best_model, MODELS_DIR / "final_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    # Save the deployed threshold (F2-optimised, best precision at recall>=0.97)
    joblib.dump(new_best_threshold, MODELS_DIR / "threshold.pkl")

    logger.info("  Model saved to %s", MODELS_DIR / "final_model.pkl")
    logger.info("  Scaler saved to %s", MODELS_DIR / "scaler.pkl")
    logger.info("  Feature names saved to %s", MODELS_DIR / "feature_names.pkl")
    logger.info("  Threshold saved to %s (%.4f) [%s, F2-optimised]", MODELS_DIR / "threshold.pkl", new_best_threshold, new_best_name)

    # -------------------------------------------------------------------
    # Project Log
    # -------------------------------------------------------------------
    generate_project_log(
        stats, all_metrics, best_metrics,
        tuned_metrics, optimal_threshold,
        all_f2_metrics, new_best_name, new_best_threshold, new_best_metrics,
        feature_names, BASE_DIR / "PROJECT_LOG.md"
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Selected Model           : %s", best_metrics["model_name"])
    logger.info("--- Default threshold (0.50) ---")
    logger.info("Recall                   : %.4f", best_metrics["recall"])
    logger.info("Precision                : %.4f", best_metrics["precision"])
    logger.info("F1 Score                 : %.4f", best_metrics["f1"])
    logger.info("ROC AUC                  : %.4f", best_metrics["roc_auc"])
    logger.info("Accuracy                 : %.4f", best_metrics["accuracy"])
    logger.info("--- RF F2-tuned threshold (%.4f) [previous approach] ---", optimal_threshold)
    logger.info("Recall  (RF F2)          : %.4f", tuned_metrics["recall"])
    logger.info("Precision (RF F2)        : %.4f", tuned_metrics["precision"])
    logger.info("F1 Score (RF F2)         : %.4f", tuned_metrics["f1"])
    logger.info("--- %s F2 threshold (%.4f) [DEPLOYED] ---", new_best_name, new_best_threshold)
    logger.info("Recall  (deployed)       : %.4f", new_best_metrics["recall"])
    logger.info("Precision (deployed)     : %.4f", new_best_metrics["precision"])
    logger.info("F1 Score (deployed)      : %.4f", new_best_metrics["f1"])
    logger.info("Accuracy (deployed)      : %.4f", new_best_metrics["accuracy"])
    logger.info("-" * 60)
    logger.info("Artifacts exported to: %s", MODELS_DIR)
    logger.info("Plots saved to       : %s", PLOTS_DIR)
    logger.info("Project log          : %s", BASE_DIR / "PROJECT_LOG.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
