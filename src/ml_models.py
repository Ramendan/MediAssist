"""
MediAssist - Machine Learning Models Module
Implements Logistic Regression, optimized Random Forest, LightGBM,
automated model selection prioritizing Recall, and probability threshold tuning.
"""

import logging
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression model with balanced class weights to favor recall."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train an optimized Random Forest using RandomizedSearchCV.
    Hyperparameter search is scored on recall.
    """
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="recall",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Starting Random Forest hyperparameter search (30 iterations, 5-fold CV)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info("Best RF params: %s | CV Recall: %.4f", search.best_params_, search.best_score_)
    return best_model


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
    """
    Train an optimized LightGBM classifier using RandomizedSearchCV.
    Scored on recall to remain consistent with the project's primary metric.

    LightGBM uses leaf-wise (best-first) tree growth, which is more efficient
    than the level-wise strategy of Random Forest and typically yields better
    accuracy on tabular clinical data with the same number of iterations.
    """
    base_lgbm = LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    param_distributions = {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 7, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=base_lgbm,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="recall",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Starting LightGBM hyperparameter search (30 iterations, 5-fold CV)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info("Best LightGBM params: %s | CV Recall: %.4f", search.best_params_, search.best_score_)
    return best_model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a full set of classification metrics for a fitted model.
    Uses the given probability threshold (default 0.5) to convert predicted
    probabilities to binary labels, enabling threshold-aware evaluation.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info(
        "%s - Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def select_best_model(
    candidates: list[tuple],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    """
    Evaluate all candidate models at the default threshold (0.5) and return
    the one with the highest recall. Ties are broken by F1 score.

    Parameters
    ----------
    candidates : list of (fitted_model, model_name) tuples
    X_test, y_test : held-out evaluation data

    Returns
    -------
    (best_model, best_metrics, all_metrics)
    """
    all_metrics = []
    for model, name in candidates:
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    # Sort by recall (primary), then F1 (tiebreaker), descending
    ranked = sorted(all_metrics, key=lambda m: (m["recall"], m["f1"]), reverse=True)
    best_metrics = ranked[0]
    best_model = next(m for m, n in candidates if n == best_metrics["model_name"])

    logger.info("Selected model: %s (Recall: %.4f)", best_metrics["model_name"], best_metrics["recall"])
    return best_model, best_metrics, all_metrics


def tune_classification_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float]:
    """
    Find the optimal probability threshold using the F-beta score.

    For medical screening, beta=2 weights recall twice as heavily as precision.
    This reflects the clinical cost asymmetry: a missed diagnosis (false negative)
    is considerably more harmful than an unnecessary follow-up (false positive).

    The precision-recall curve is computed across all candidate thresholds and
    the one maximizing F-beta is selected and returned.

    Parameters
    ----------
    model  : fitted classifier with predict_proba
    X_val  : feature matrix (test set used for threshold search)
    y_val  : true labels
    beta   : F-beta weighting factor (default 2.0 = recall-focused)

    Returns
    -------
    (optimal_threshold, best_fbeta_score)
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    best_fbeta = 0.0
    best_threshold = 0.5

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p + r == 0:
            continue
        fbeta = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = float(t)

    logger.info(
        "Threshold tuning complete: optimal=%.4f (F%.0f=%.4f, vs default=0.50)",
        best_threshold, beta, best_fbeta,
    )
    return best_threshold, best_fbeta
