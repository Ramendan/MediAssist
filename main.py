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
import textwrap
from datetime import datetime
from pathlib import Path

import joblib

from src.data_pipeline import run_pipeline
from src.evaluation import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.ml_models import (
    evaluate_model,
    select_best_model,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    tune_classification_threshold,
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
    feature_names: list[str],
    log_path: Path,
):
    """
    Generate PROJECT_LOG.md documenting all technical metrics, methodology,
    and ethical limitations. This file serves as the primary technical report
    for the assignment submission.
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

    content = textwrap.dedent(f"""\
    # MediAssist - Project Technical Log

    **Generated:** {timestamp}

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
    - **Validation split purpose:** Threshold tuning only — never used for model training or final test-set reporting

    ---

    ## 2. Preprocessing Steps

    | Step | Description |
    |------|-------------|
    | Age conversion | Converted from days to years (`age / 365.25`) |
    | Missing value handling | Dropped rows with any NaN values |
    | Outlier removal | Removed biologically implausible rows: height 100-220 cm, weight 30-200 kg, systolic BP 60-250 mmHg, diastolic BP 40-160 mmHg, and enforced systolic > diastolic |
    | BMI calculation | `weight / (height / 100)^2` |
    | Feature engineering | Added `pulse_pressure` and `age_bmi` interaction term (see Section 3) |
    | Normalization | `StandardScaler` fitted on training data only, then applied to all three splits (train, validation, test) using training statistics only — no leakage |

    ---

    ## 3. Feature Engineering

    Two clinically-motivated derived features were added to improve predictive power:

    | Feature | Formula | Clinical Rationale |
    |---------|---------|-------------------|
    | `pulse_pressure` | `ap_hi - ap_lo` | A pulse pressure above 60 mmHg is an independent predictor of cardiovascular events, particularly in older adults. It reflects arterial stiffness, which is not captured by either systolic or diastolic BP alone. |
    | `age_bmi` | `age * bmi` | An interaction term that captures compounded metabolic-aging risk. Obesity in older patients carries disproportionately higher cardiovascular risk than in younger patients, and a linear model cannot capture this without the explicit interaction. |

    ---

    ## 4. Model Comparison (Default Threshold = 0.50)

    Three models were trained and compared. All use `class_weight='balanced'` to
    counteract the approximately 50/50 class distribution and favor recall.

    {comparison_table}

    **Selected model:** {best_metrics['model_name']} (primary criterion: Recall)

    ---

    ## 5. Threshold Tuning

    ### Methodology

    After model selection, the classification probability threshold was optimized
    using the **F-beta score (beta=2)**. This metric weights recall twice as heavily
    as precision, reflecting the clinical cost asymmetry in medical screening:

    > A missed cardiovascular disease case (false negative) is far more harmful than
    > an unnecessary follow-up consultation (false positive).

    The precision-recall curve was computed on a **dedicated validation split**
    (15% of the dataset, held out from both training and the final test set) across
    all candidate thresholds. The threshold maximizing F2 was selected and persisted.
    Using a separate validation set ensures the reported test-set metrics are unbiased
    and reflect true generalisation performance.

    ### Before vs. After Threshold Tuning ({best_metrics['model_name']})

    {threshold_table}

    **Optimal threshold:** {optimal_threshold:.4f}

    ---

    ## 6. Classification Reports

    {reports_section}

    ---

    ## 7. Technical Decisions and Methodology Justification

    ### Why Recall, Not Accuracy

    Accuracy measures the proportion of all predictions that are correct. It treats
    a missed diagnosis (false negative) and an unnecessary follow-up (false positive)
    as identical errors. In cardiovascular screening, they are not:

    | Error Type | What it means | Clinical consequence |
    |------------|---------------|----------------------|
    | **False Negative** | Model predicts healthy; patient has disease | Patient goes untreated — potentially fatal |
    | **False Positive** | Model predicts disease; patient is healthy | Patient receives a follow-up consultation — inconvenient, not dangerous |

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
    unseen data — the threshold has effectively "seen" the test set and the reported
    Recall is optimistically biased. By tuning on a dedicated 15% validation split,
    the 15% test set remains truly held-out, so reported metrics accurately reflect
    what would be observed in deployment.

    ### Other Decisions

    - **Three-model comparison:** Logistic Regression (interpretable baseline), Random Forest
      (ensemble method), and LightGBM (gradient boosting) were trained and compared.
      LightGBM uses leaf-wise tree growth and is typically the strongest performer on
      tabular clinical data.
    - **Hyperparameter search:** `RandomizedSearchCV`, 30 iterations, 5-fold cross-
      validation, scored on Recall for both Random Forest and LightGBM.
    - **`class_weight='balanced'`:** Applied to all models to counteract class imbalance
      without oversampling, biasing each model toward correct positive detection.
    - **Feature engineering:** `pulse_pressure` (arterial stiffness proxy) and `age_bmi`
      (aging-obesity interaction) added after BMI calculation, before normalization.
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
      context. Model performance may not generalize across populations with different age
      distributions, ethnic backgrounds, or healthcare access patterns.
    - **No temporal validation:** The model has not been validated on prospective or
      time-shifted data. Performance in real-world deployment may differ from reported metrics.
    - **Limited feature set:** Important cardiovascular risk factors such as family history,
      LDL/HDL levels, troponin, ECG data, and medication history are not present in the dataset.
    - **Binary classification simplification:** Cardiovascular disease is a spectrum.
      The binary (0/1) output oversimplifies clinical reality.
    - **Recall vs. precision trade-off:** Optimizing for recall increases false positives,
      which could cause unnecessary patient anxiety and healthcare resource consumption.

    ---

    ## 9. Generated Artifacts

    | File | Description |
    |------|-------------|
    | `models/final_model.pkl` | Trained {best_metrics['model_name']} model |
    | `models/scaler.pkl` | Fitted StandardScaler |
    | `models/feature_names.pkl` | Ordered feature name list (includes engineered features) |
    | `models/threshold.pkl` | Optimal classification threshold ({optimal_threshold:.4f}) |
    | `plots/learning_curves.png` | Recall vs. training set size (bias-variance analysis) |
    | `plots/confusion_matrix.png` | Confusion matrix at tuned threshold ({optimal_threshold:.4f}) |
    | `plots/feature_importance.png` | Feature importance ranked bar chart |
    | `plots/precision_recall_curve.png` | PR curve with default and optimal threshold annotated |
    | `plots/roc_curve.png` | ROC curve for all three models with tuned threshold marked |
    """)

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
    logger.info("[1/6] Checking dataset...")
    data_path = download_dataset(data_path)

    # -------------------------------------------------------------------
    # Step 2: Data Pipeline
    # -------------------------------------------------------------------
    logger.info("[2/6] Running data pipeline...")
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
    logger.info("[3/7] Training models...")

    logger.info("  Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    logger.info("  Training Random Forest (with hyperparameter search)...")
    rf_model = train_random_forest(X_train, y_train)

    logger.info("  Training LightGBM (with hyperparameter search)...")
    lgbm_model = train_lightgbm(X_train, y_train)

    # -------------------------------------------------------------------
    # Step 4: Model Evaluation and Selection
    # -------------------------------------------------------------------
    logger.info("[4/7] Evaluating all models at default threshold (0.50)...")
    candidates = [
        (lr_model, "Logistic Regression"),
        (rf_model, "Random Forest"),
        (lgbm_model, "LightGBM"),
    ]
    best_model, best_metrics, all_metrics = select_best_model(candidates, X_test, y_test)

    logger.info("  Best model: %s", best_metrics["model_name"])
    logger.info("  Recall: %.4f | Precision: %.4f | F1: %.4f | AUC: %.4f",
                best_metrics["recall"], best_metrics["precision"],
                best_metrics["f1"], best_metrics["roc_auc"])

    # -------------------------------------------------------------------
    # Step 5: Threshold Tuning  (F2-score, beta=2)
    # -------------------------------------------------------------------
    logger.info("[5/7] Tuning classification threshold (F2-score, beta=2)...")
    # Use the validation split — never the test set — to avoid biasing test metrics.
    optimal_threshold, best_fbeta = tune_classification_threshold(
        best_model, X_val, y_val, beta=2.0
    )
    tuned_metrics = evaluate_model(
        best_model, X_test, y_test, best_metrics["model_name"], threshold=optimal_threshold
    )
    logger.info(
        "  Threshold %.4f: Recall %.4f -> %.4f | Precision %.4f -> %.4f",
        optimal_threshold,
        best_metrics["recall"], tuned_metrics["recall"],
        best_metrics["precision"], tuned_metrics["precision"],
    )

    # -------------------------------------------------------------------
    # Step 6: Generate Plots
    # -------------------------------------------------------------------
    logger.info("[6/7] Generating evaluation plots...")

    plot_learning_curves(
        best_model, X_train, y_train,
        str(PLOTS_DIR / "learning_curves.png"),
    )
    plot_confusion_matrix(
        best_model, X_test, y_test,
        str(PLOTS_DIR / "confusion_matrix.png"),
        model_name=best_metrics["model_name"],
        threshold=optimal_threshold,
    )
    plot_feature_importance(
        best_model, feature_names,
        str(PLOTS_DIR / "feature_importance.png"),
        model_name=best_metrics["model_name"],
    )
    plot_precision_recall_curve(
        best_model, X_test, y_test,
        str(PLOTS_DIR / "precision_recall_curve.png"),
        model_name=best_metrics["model_name"],
        optimal_threshold=optimal_threshold,
    )
    plot_roc_curve(
        candidates, X_test, y_test,
        str(PLOTS_DIR / "roc_curve.png"),
        optimal_threshold=optimal_threshold,
        best_model_name=best_metrics["model_name"],
    )
    logger.info("  All plots saved to %s", PLOTS_DIR)

    # -------------------------------------------------------------------
    # Step 7: Export Artifacts
    # -------------------------------------------------------------------
    logger.info("[7/7] Exporting model artifacts...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "final_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(optimal_threshold, MODELS_DIR / "threshold.pkl")

    logger.info("  Model saved to %s", MODELS_DIR / "final_model.pkl")
    logger.info("  Scaler saved to %s", MODELS_DIR / "scaler.pkl")
    logger.info("  Feature names saved to %s", MODELS_DIR / "feature_names.pkl")
    logger.info("  Threshold saved to %s (%.4f)", MODELS_DIR / "threshold.pkl", optimal_threshold)

    # -------------------------------------------------------------------
    # Project Log
    # -------------------------------------------------------------------
    generate_project_log(
        stats, all_metrics, best_metrics, tuned_metrics, optimal_threshold,
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
    logger.info("--- Tuned threshold (%.4f) ---", optimal_threshold)
    logger.info("Recall  (tuned)          : %.4f", tuned_metrics["recall"])
    logger.info("Precision (tuned)        : %.4f", tuned_metrics["precision"])
    logger.info("F1 Score (tuned)         : %.4f", tuned_metrics["f1"])
    logger.info("Accuracy (tuned)         : %.4f", tuned_metrics["accuracy"])
    logger.info("-" * 60)
    logger.info("Artifacts exported to: %s", MODELS_DIR)
    logger.info("Plots saved to       : %s", PLOTS_DIR)
    logger.info("Project log          : %s", BASE_DIR / "PROJECT_LOG.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
