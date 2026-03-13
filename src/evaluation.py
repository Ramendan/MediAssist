"""
MediAssist - Evaluation and Visualization Module
Generates learning curves, confusion matrix, feature importance,
precision-recall curve, and ROC curve plots.
All plots use a professional, clean aesthetic with no emojis.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Professional styling
plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "grid.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.size": 11,
})


def plot_learning_curves(model, X: np.ndarray, y: np.ndarray, save_path: str) -> str:
    """
    Generate and save learning curves showing training and cross-validation scores.
    Uses recall as the scoring metric.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2196F3")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#FF5722")

    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", linewidth=2, markersize=5, label="Training Recall")
    ax.plot(train_sizes, val_mean, "o-", color="#FF5722", linewidth=2, markersize=5, label="Cross-Validation Recall")

    final_val = val_mean[-1]
    final_train = train_mean[-1]
    gap = final_train - final_val
    gap_label = f"Generalisation gap at full size: {gap:.3f}"

    ax.set_title("Learning Curves (Recall)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Recall Score", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.04, gap_label, transform=ax.transAxes,
        fontsize=10, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Learning curves saved to %s", save_path)
    return save_path


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    model_name: str = "Best Model",
    threshold: float = 0.5,
) -> str:
    """
    Generate and save a confusion matrix with recall, precision, and F1 annotations.

    Applies the given probability threshold (default: 0.5) to convert predicted
    probabilities to binary labels. Using the tuned threshold here ensures the
    saved plot accurately reflects the model's real-world screening behaviour.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
        linewidths=0.5,
        linecolor="#cccccc",
        annot_kws={"size": 16, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name} (threshold={threshold:.2f})", fontsize=14, fontweight="bold", pad=15)

    metrics_text = f"Recall: {recall:.4f}  |  Precision: {precision:.4f}  |  F1: {f1:.4f}  |  Threshold: {threshold:.2f}"
    ax.text(
        0.5,
        -0.12,
        metrics_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#90caf9"),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved to %s", save_path)
    return save_path


def plot_feature_importance(
    model,
    feature_names: list[str],
    save_path: str,
    model_name: str = "Best Model",
) -> str:
    """
    Generate and save a horizontal bar chart of feature importances.
    Supports both tree-based (.feature_importances_) and linear (.coef_) models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Feature Importance (Gini)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient Magnitude"
    else:
        logger.warning("Model does not expose feature importances or coefficients.")
        return ""

    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.45)))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_importances)))
    ax.barh(sorted_names, sorted_importances, color=colors, edgecolor="#ffffff", height=0.7)

    for i, (val, name) in enumerate(zip(sorted_importances, sorted_names)):
        ax.text(val + max(sorted_importances) * 0.01, i, f"{val:.4f}", va="center", fontsize=10, color="#333333")

    ax.set_xlabel(importance_label, fontsize=12)
    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance chart saved to %s", save_path)
    return save_path


def plot_precision_recall_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    model_name: str = "Best Model",
    optimal_threshold: float = 0.5,
) -> str:
    """
    Generate and save a Precision-Recall curve with threshold annotations.

    Two reference points are marked:
    - Default threshold (0.50): baseline operating point.
    - Optimal threshold: the F2-score-maximizing point selected during tuning.

    The area under the PR curve (AP score) is displayed in the legend.
    A high AP score on an imbalanced dataset is more informative than ROC AUC.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recalls, precisions)

    # Get operating point metrics at each reference threshold
    def _point_at_threshold(t):
        idx = np.argmin(np.abs(thresholds - t))
        return precisions[idx], recalls[idx]

    p_default, r_default = _point_at_threshold(0.5)
    p_optimal, r_optimal = _point_at_threshold(optimal_threshold)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        recalls, precisions,
        color="#2196F3", linewidth=2.5,
        label=f"Precision-Recall Curve (AP = {pr_auc:.4f})",
    )

    ax.scatter(
        r_default, p_default,
        s=120, zorder=5, color="#9C27B0",
        label=f"Default threshold (0.50) — P={p_default:.3f}, R={r_default:.3f}",
        marker="D",
    )
    ax.scatter(
        r_optimal, p_optimal,
        s=160, zorder=5, color="#FF5722",
        label=f"Optimal threshold ({optimal_threshold:.2f}) — P={p_optimal:.3f}, R={r_optimal:.3f}",
        marker="*",
    )

    # Annotate baseline (random classifier) line
    baseline = float(np.sum(y_test) / len(y_test))
    ax.axhline(y=baseline, color="#aaaaaa", linestyle="--", linewidth=1, label=f"Baseline (random) = {baseline:.3f}")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Precision-recall curve saved to %s", save_path)
    return save_path


def plot_roc_curve(
    models_and_names: list[tuple],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    optimal_threshold: float = 0.5,
    best_model_name: str = "",
) -> str:
    """
    Generate and save a ROC curve for all trained models on the same axes.

    Plotting all three models (LR, RF, LightGBM) together provides an honest
    comparison — the selected model is highlighted. The optimal operating point
    (tuned threshold) is annotated on the selected model's curve.

    Parameters
    ----------
    models_and_names   : list of (fitted_model, model_name) tuples
    X_test, y_test     : held-out test arrays
    save_path          : output file path
    optimal_threshold  : F2-tuned threshold — annotated on the best model's curve
    best_model_name    : name of the selected model (highlighted with a thicker line)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (model, name) in enumerate(models_and_names):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        model_auc = roc_auc_score(y_test, y_prob)

        is_best = name == best_model_name
        lw = 3.0 if is_best else 1.5
        alpha = 1.0 if is_best else 0.6
        label = f"{name} (AUC = {model_auc:.4f})" + (" ← selected" if is_best else "")

        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=lw, alpha=alpha, label=label)

        # Annotate the tuned threshold operating point on the selected model only
        if is_best:
            y_prob_thresh = (y_prob >= optimal_threshold).astype(int)
            from sklearn.metrics import confusion_matrix as _cm
            tn, fp, fn, tp = _cm(y_test, y_prob_thresh).ravel()
            op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ax.scatter(
                op_fpr, op_tpr,
                s=160, zorder=6, color="#FF5722", marker="*",
                label=f"Tuned threshold ({optimal_threshold:.2f}) — TPR={op_tpr:.3f}, FPR={op_fpr:.3f}",
            )

    # Diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=1, label="Random classifier (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curve — All Models", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC curve saved to %s", save_path)
    return save_path

