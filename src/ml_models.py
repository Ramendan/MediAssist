"""
MediAssist - Machine Learning Models Module
Implements Logistic Regression, optimized Random Forest, LightGBM, XGBoost,
automated model selection prioritizing Recall, F2-score threshold tuning,
and precision-constrained threshold tuning (recall floor strategy).
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
from xgboost import XGBClassifier

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

class SoftVotingEnsemble:
    """
    Inference-only soft voting ensemble that averages predict_proba outputs
    from a list of already-fitted models. No additional training is required.

    Averaging probabilities from diverse models (RF, LightGBM, XGBoost) reduces
    individual model variance, produces better-calibrated probability scores,
    and raises the Precision-Recall curve ceiling compared to any single model.
    An improved PR curve means: at the same high recall (e.g. 97%), the ensemble
    can achieve higher precision than any one component.
    """

    def __init__(self, estimators: list, name: str = "Soft Voting Ensemble"):
        self.estimators_ = estimators
        self._name = name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """Average feature importances from tree-based members."""
        importances = [
            m.feature_importances_
            for m in self.estimators_
            if hasattr(m, "feature_importances_")
        ]
        if importances:
            return np.mean(importances, axis=0)
        raise AttributeError("No member with feature_importances_")

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


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Train an optimized XGBoost classifier using RandomizedSearchCV.
    Scored on recall for consistency with the project's primary metric.

    XGBoost uses `scale_pos_weight` to handle class imbalance (analogous to
    `class_weight='balanced'` in sklearn estimators). Its regularisation
    parameters (gamma, lambda, alpha) and level-wise tree growth often produce
    better-calibrated probabilities than LightGBM on medical tabular data,
    which translates to a higher-precision operating point at any given recall.
    """
    # Compute class imbalance ratio from training labels
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    base_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )

    param_distributions = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 0.5],
    }

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="recall",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Starting XGBoost hyperparameter search (30 iterations, 5-fold CV)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info("Best XGBoost params: %s | CV Recall: %.4f", search.best_params_, search.best_score_)
    return best_model


class SoftVotingEnsemble:
    """
    Inference-only soft voting ensemble that averages predict_proba outputs
    from a list of already-fitted models. No additional training is required.

    Averaging probabilities from diverse models (RF, LightGBM, XGBoost) reduces
    individual model variance, produces better-calibrated probability scores,
    and raises the Precision-Recall curve ceiling compared to any single model.
    An improved PR curve means: at the same high recall (~97%), the ensemble
    can achieve higher precision than any one component alone.
    """

    def __init__(self, estimators: list, name: str = "Soft Voting Ensemble"):
        self.estimators_ = estimators
        self._name = name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """Average feature importances from tree-based members."""
        importances = [
            m.feature_importances_
            for m in self.estimators_
            if hasattr(m, "feature_importances_")
        ]
        if importances:
            return np.mean(importances, axis=0)
        raise AttributeError("No ensemble member exposes feature_importances_")


def tune_threshold_recall_floor(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    recall_floor: float = 0.90,
) -> tuple[float, float, float]:
    """
    Find the classification threshold that maximises precision while keeping
    recall at or above `recall_floor`.

    This is the precision-constrained alternative to F2 threshold tuning.
    Instead of pushing recall to ~0.98 at the cost of precision (~0.52), this
    strategy anchors recall at a clinically acceptable minimum (default 0.90,
    meaning at most 10% of true disease cases are missed) and then finds the
    highest possible precision within that constraint.

    The result is a better-balanced operating point, e.g. Recall=0.90,
    Precision≈0.65, which reduces unnecessary false-positive follow-ups while
    still catching the vast majority of true disease cases.

    Parameters
    ----------
    model        : fitted classifier with predict_proba
    X_val        : feature matrix (validation split — never the test set)
    y_val        : true labels
    recall_floor : minimum acceptable recall (default 0.90)

    Returns
    -------
    (optimal_threshold, precision_at_threshold, recall_at_threshold)
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    # Find all (p, r, t) triplets where recall meets the floor constraint
    eligible = [
        (float(p), float(r), float(t))
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds)
        if r >= recall_floor
    ]

    if eligible:
        # Among eligible thresholds, choose the one with highest precision
        best_p, best_r, best_t = max(eligible, key=lambda x: x[0])
    else:
        # Fallback: no threshold achieves the floor — return the max-recall point
        best_idx = int(np.argmax(recalls[:-1]))
        best_t = float(thresholds[best_idx])
        best_p = float(precisions[best_idx])
        best_r = float(recalls[best_idx])
        logger.warning(
            "No threshold achieves recall_floor=%.2f; falling back to max-recall "
            "threshold=%.4f (Recall=%.4f, Precision=%.4f)",
            recall_floor, best_t, best_r, best_p,
        )

    logger.info(
        "Constrained threshold (recall_floor=%.2f): threshold=%.4f | P=%.4f | R=%.4f",
        recall_floor, best_t, best_p, best_r,
    )
    return best_t, best_p, best_r


def select_best_model_by_f2(
    candidates: list[tuple],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    beta: float = 2.0,
    min_recall: float = 0.97,
) -> tuple:
    """
    For each candidate model (including SoftVotingEnsemble), tune the F-beta
    threshold on the validation set and evaluate on the test set. Returns the
    model that achieves the highest precision while keeping recall >= min_recall.

    This answers: 'Can any candidate get ~97%+ recall with *better* precision
    than the single best model alone?' A soft-voting ensemble typically surfaces
    here because its averaged probabilities have a higher-ceiling PR curve.

    Parameters
    ----------
    candidates  : list of (fitted_model, model_name) tuples
    X_val       : validation features (threshold tuning only — no test leakage)
    y_val       : validation labels
    X_test      : test features (final evaluation only)
    y_test      : test labels
    beta        : F-beta weight (default 2.0)
    min_recall  : minimum acceptable test-set recall at the F2 threshold

    Returns
    -------
    (best_model, best_name, best_threshold, best_metrics, all_f2_metrics)
        all_f2_metrics  list of metric dicts, one per candidate, each containing
                        opt_threshold and opt_fbeta alongside standard metrics
    """
    all_results = []
    for model, name in candidates:
        threshold, fbeta = tune_classification_threshold(model, X_val, y_val, beta=beta)
        metrics = evaluate_model(model, X_test, y_test, name, threshold=threshold)
        metrics["opt_threshold"] = threshold
        metrics["opt_fbeta"] = fbeta
        all_results.append((model, name, threshold, metrics))
        logger.info(
            "  [F2 selection] %-24s | threshold=%.4f | Recall=%.4f | Precision=%.4f",
            name, threshold, metrics["recall"], metrics["precision"],
        )

    # Among those with test-set recall >= min_recall, pick highest precision
    eligible = [r for r in all_results if r[3]["recall"] >= min_recall]
    if eligible:
        best = max(eligible, key=lambda r: r[3]["precision"])
        logger.info(
            "Best F2 model (max precision at recall>=%.2f): %s | P=%.4f | R=%.4f",
            min_recall, best[1], best[3]["precision"], best[3]["recall"],
        )
    else:
        best = max(all_results, key=lambda r: r[3]["recall"])
        logger.warning(
            "No candidate reached recall>=%.2f; fallback to max-recall: %s (R=%.4f)",
            min_recall, best[1], best[3]["recall"],
        )

    best_model, best_name, best_threshold, best_metrics = best
    all_f2_metrics = [r[3] for r in all_results]
    return best_model, best_name, best_threshold, best_metrics, all_f2_metrics


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
