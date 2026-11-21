"""
Threshold Calibration Engine for Prompt Injection Firewalls.

Sweeps decision thresholds on validation data to achieve a strict target False Positive Rate
(e.g., target FPR <= 1%) while optimizing Recall and F1 score.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def calibrate_decision_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    target_fpr: float = 0.01,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    steps: int = 100,
) -> Dict[str, Any]:
    """
    Finds the optimal decision threshold on validation data.
    
    Args:
        y_val: Ground truth binary labels (0=benign, 1=malicious).
        probs_val: Predicted injection probabilities for validation set.
        target_fpr: Desired upper bound for False Positive Rate (default 1%).
    """
    y_val = np.asarray(y_val, dtype=int)
    probs_val = np.asarray(probs_val, dtype=float)

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    results = []

    best_calibrated_threshold = 0.5
    found_target = False

    for t in thresholds:
        preds = (probs_val >= t).astype(int)

        tp = np.sum((y_val == 1) & (preds == 1))
        fp = np.sum((y_val == 0) & (preds == 1))
        tn = np.sum((y_val == 0) & (preds == 0))
        fn = np.sum((y_val == 1) & (preds == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        entry = {
            "threshold": round(float(t), 3),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "fp_count": int(fp),
            "tp_count": int(tp),
        }
        results.append(entry)

        if fpr <= target_fpr and not found_target:
            best_calibrated_threshold = float(t)
            found_target = True

    if not found_target:
        # Fallback to threshold that yields lowest FPR
        results.sort(key=lambda x: x["fpr"])
        best_calibrated_threshold = results[0]["threshold"]

    return {
        "calibrated_threshold": round(best_calibrated_threshold, 3),
        "target_fpr": target_fpr,
        "achieved_validation_fpr": next((r["fpr"] for r in results if r["threshold"] == round(best_calibrated_threshold, 3)), None),
        "validation_f1_at_threshold": next((r["f1"] for r in results if r["threshold"] == round(best_calibrated_threshold, 3)), None),
        "sweep_curve": results,
    }
