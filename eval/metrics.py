"""
Core metric calculation engine for Prompt Injection Firewall evaluation.

Computes:
- Precision, Recall, F1 Score, Accuracy
- False Positive Rate (FPR) = FP / (FP + TN)
- ROC-AUC and PR-AUC
- Non-parametric bootstrap 95% confidence intervals (for small-sample robustness)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class ClassificationMetrics:
    """Stores full evaluation metrics for a binary classifier."""
    threshold: float
    total_samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: float
    roc_auc: float
    pr_auc: float
    ci_f1: Optional[Tuple[float, float]] = None
    ci_fpr: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Computes binary classification metrics at a given decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ROC AUC and PR AUC require both classes to be present in y_true
    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_probs)) if has_both_classes else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if has_both_classes else 0.0

    return ClassificationMetrics(
        threshold=round(threshold, 4),
        total_samples=len(y_true),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        fpr=round(fpr, 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
    )


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric_name: str = "f1",
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes empirical bootstrap confidence interval for a specified metric.
    
    Returns:
        (ci_lower, ci_upper) at (1 - alpha) confidence (default 95%).
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    n = len(y_true)

    if n == 0:
        return (0.0, 0.0)

    bootstrap_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        sample_true = y_true[indices]
        sample_probs = y_probs[indices]
        sample_pred = (sample_probs >= threshold).astype(int)

        if metric_name == "f1":
            tp = np.sum((sample_true == 1) & (sample_pred == 1))
            fp = np.sum((sample_true == 0) & (sample_pred == 1))
            fn = np.sum((sample_true == 1) & (sample_pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        elif metric_name == "fpr":
            fp = np.sum((sample_true == 0) & (sample_pred == 1))
            tn = np.sum((sample_true == 0) & (sample_pred == 0))
            val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        elif metric_name == "precision":
            tp = np.sum((sample_true == 1) & (sample_pred == 1))
            fp = np.sum((sample_true == 0) & (sample_pred == 1))
            val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric_name == "recall":
            tp = np.sum((sample_true == 1) & (sample_pred == 1))
            fn = np.sum((sample_true == 1) & (sample_pred == 0))
            val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            raise ValueError(f"Unsupported metric for bootstrap: {metric_name}")

        bootstrap_scores.append(val)

    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100
    ci_low = float(np.percentile(bootstrap_scores, lower_percentile))
    ci_high = float(np.percentile(bootstrap_scores, upper_percentile))

    return (round(ci_low, 4), round(ci_high, 4))
