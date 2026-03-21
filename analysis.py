"""
Reusable analysis functions for CS421 Anomaly Detection report.

Usage in any model notebook:
    from analysis import (
        model_metrics, plot_score_distribution, plot_roc_curve,
        plot_failure_analysis, plot_feature_importance, log_results, load_results,
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

RESULTS_DIR = "results"
SCORES_PATH = os.path.join(RESULTS_DIR, "scores.csv")


# ── Evaluation ────────────────────────────────────────────────────────────────

def model_metrics(
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute all metrics for a model's predictions on the test set.
    Returns a dict with AUC, Precision, Recall, F1 (at best-F1 threshold).
    """
    scores_norm = _normalise(scores)

    # Find threshold that maximises F1
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.01, 0.99, 500):
        preds = (scores_norm >= t).astype(int)
        f = f1_score(test_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds = (scores_norm >= best_t).astype(int)
    metrics = {
        "model": model_name,
        "AUC": roc_auc_score(test_labels, scores),
        "Precision": precision_score(test_labels, preds, zero_division=0),
        "Recall": recall_score(test_labels, preds, zero_division=0),
        "F1": best_f1,
        "threshold": best_t,
    }

    print(f"{model_name}")
    print(f"# AUC:       {metrics['AUC']:.4f}")
    print(f"# Precision: {metrics['Precision']:.4f}")
    print(f"# Recall:    {metrics['Recall']:.4f}")
    print(f"# F1 Score:  {metrics['F1']:.4f}")
    return metrics


# ── Score Distribution ────────────────────────────────────────────────────────

def plot_score_distribution(
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    batch_name: str,
):
    """Plot KDE of anomaly scores for normal vs anomalous users."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(scores[test_labels == 0], label="Normal", fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(scores[test_labels == 1], label="Anomalous", fill=True, alpha=0.5, ax=ax)
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"Score Distribution — {model_name} ({batch_name})")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    batch_name: str,
):
    """Plot ROC curve for a single model."""
    fpr, tpr, _ = roc_curve(test_labels, scores)
    auc_val = roc_auc_score(test_labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name} ({batch_name})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ── Precision-Recall Curve ────────────────────────────────────────────────────

def plot_pr_curve(
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    batch_name: str,
):
    """Plot Precision-Recall curve — more informative than ROC for imbalanced data."""
    precision, recall, _ = precision_recall_curve(test_labels, scores)
    ap = average_precision_score(test_labels, scores)
    baseline = test_labels.sum() / len(test_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.3, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name} ({batch_name})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    batch_name: str,
):
    """Plot confusion matrix at the best-F1 threshold."""
    scores_norm = _normalise(scores)

    # Find threshold that maximises F1
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.01, 0.99, 500):
        preds = (scores_norm >= t).astype(int)
        f = f1_score(test_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds = (scores_norm >= best_t).astype(int)
    cm = confusion_matrix(test_labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name} ({batch_name})\nthreshold={best_t:.3f}, F1={best_f1:.3f}")
    plt.tight_layout()
    plt.show()


# ── Failure Case Analysis ────────────────────────────────────────────────────

def plot_failure_analysis(
    test_df: pd.DataFrame,
    test_labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    feature_cols: list[str],
    compare_feats: list[str] | None = None,
):
    """
    Identify false negatives/positives and compare their feature distributions.
    """
    if compare_feats is None:
        compare_feats = [
            "prop_extreme", "abs_mean_deviation", "item_coverage_ratio",
            "rating_entropy", "rating_std", "avg_item_avg_rating",
        ]
    # filter to features that exist
    compare_feats = [f for f in compare_feats if f in feature_cols]

    scores_norm = _normalise(scores)
    threshold = np.median(scores_norm[test_labels == 1])
    preds = (scores_norm >= threshold).astype(int)

    df = test_df[feature_cols].copy()
    df["label"] = test_labels
    df["predicted"] = preds

    tp = df[(df["label"] == 1) & (df["predicted"] == 1)]
    fn = df[(df["label"] == 1) & (df["predicted"] == 0)]
    fp = df[(df["label"] == 0) & (df["predicted"] == 1)]

    print(f"  True positives:  {len(tp)}")
    print(f"  False negatives: {len(fn)} (missed anomalies)")
    print(f"  False positives: {len(fp)} (false alarms)")

    if len(fn) == 0:
        print("  No false negatives — nothing to compare.")
        return

    n_feats = len(compare_feats)
    ncols = min(3, n_feats)
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_feats == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, feat in zip(axes, compare_feats):
        if len(tp) > 0:
            sns.kdeplot(tp[feat], label="Caught (TP)", fill=True, ax=ax, alpha=0.4)
        sns.kdeplot(fn[feat], label="Missed (FN)", fill=True, ax=ax, alpha=0.4)
        ax.set_title(feat)
        ax.legend(fontsize=8)

    for j in range(len(compare_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Caught vs Missed Anomalies — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_cols: list[str],
    model_name: str,
):
    """Plot horizontal bar chart of feature importances."""
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.35)))
    feat_imp.plot.barh(ax=ax)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()


# ── Results Logging ───────────────────────────────────────────────────────────

def log_results(
    model_name: str,
    batch_name: str,
    metrics: dict,
    train_time: float | None = None,
    cv_auc: float | None = None,
    codabench_auc: float | None = None,
    notes: str = "",
) -> pd.DataFrame:
    """
    Append a row to results/scores.csv. Deduplicates by model+batch+date.
    Returns the full scores DataFrame.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    row = {
        "date": date.today().isoformat(),
        "model": model_name,
        "batch": batch_name,
        "cv_auc": cv_auc,
        "local_auc": metrics.get("AUC"),
        "precision": metrics.get("Precision"),
        "recall": metrics.get("Recall"),
        "f1": metrics.get("F1"),
        "codabench_auc": codabench_auc,
        "train_time_s": round(train_time, 2) if train_time else None,
        "notes": notes,
    }
    new_df = pd.DataFrame([row])

    if os.path.exists(SCORES_PATH):
        existing = pd.read_csv(SCORES_PATH)
        # drop same model+batch+date to avoid duplicates
        mask = (
            (existing["model"] == model_name) &
            (existing["batch"] == batch_name) &
            (existing["date"] == row["date"])
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(SCORES_PATH, index=False)
    print(f"Logged to {SCORES_PATH}")
    return combined


def load_results() -> pd.DataFrame:
    """Load the full results history."""
    if os.path.exists(SCORES_PATH):
        return pd.read_csv(SCORES_PATH)
    return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
