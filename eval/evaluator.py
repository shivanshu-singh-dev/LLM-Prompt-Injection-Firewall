"""
Standard Evaluation Suite for Prompt Injection Classifiers.

Evaluates any detector/model across:
1. Held-out random test split (Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, 95% CI)
2. Per-attack-family detection rates (role_override, jailbreak, indirect, obedience_exploit)
3. Hard-negative false positive rate (reported independently to measure real-world usability)
"""

from typing import List, Dict, Any, Callable, Optional, Union
import json
from pathlib import Path
import numpy as np
from data.schema import PromptExample
from eval.metrics import compute_binary_metrics, compute_bootstrap_ci, ClassificationMetrics


class FirewallEvaluator:
    """Evaluates classifier predictions on test and hard negative datasets."""

    def __init__(
        self,
        test_examples: List[PromptExample],
        hard_negatives: List[PromptExample],
    ):
        self.test_examples = test_examples
        self.hard_negatives = hard_negatives

    @classmethod
    def from_files(
        cls,
        test_file: str = "data/processed/test.jsonl",
        hard_neg_file: str = "data/processed/hard_negatives.jsonl",
    ) -> "FirewallEvaluator":
        """Loads evaluation data directly from JSONL paths."""
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test split not found at {test_path}")

        with open(test_path, "r", encoding="utf-8") as f:
            test_examples = [PromptExample.from_json(line) for line in f if line.strip()]

        hard_neg_path = Path(hard_neg_file)
        if hard_neg_path.exists():
            with open(hard_neg_path, "r", encoding="utf-8") as f:
                hard_negatives = [PromptExample.from_json(line) for line in f if line.strip()]
        else:
            hard_negatives = []

        return cls(test_examples=test_examples, hard_negatives=hard_negatives)

    def evaluate(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        threshold: float = 0.5,
        model_name: str = "Classifier",
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation.
        
        Args:
            predict_fn: Function that accepts List[str] and returns 1D array of injection probabilities [0.0, 1.0].
            threshold: Decision threshold.
            model_name: Display name for the model.
        """
        # 1. Evaluate on Held-out Test Split
        test_texts = [ex.text for ex in self.test_examples]
        test_labels = np.array([ex.label for ex in self.test_examples], dtype=int)
        test_probs = np.array(predict_fn(test_texts), dtype=float)

        base_metrics = compute_binary_metrics(test_labels, test_probs, threshold=threshold)
        ci_f1 = compute_bootstrap_ci(test_labels, test_probs, metric_name="f1", threshold=threshold)
        ci_fpr = compute_bootstrap_ci(test_labels, test_probs, metric_name="fpr", threshold=threshold)
        base_metrics.ci_f1 = ci_f1
        base_metrics.ci_fpr = ci_fpr

        # 2. Per-Family Breakdown on Test Split
        family_results: Dict[str, Dict[str, Any]] = {}
        unique_families = sorted(list(set(ex.attack_family for ex in self.test_examples)))

        for family in unique_families:
            fam_indices = [i for i, ex in enumerate(self.test_examples) if ex.attack_family == family]
            if not fam_indices:
                continue

            fam_labels = test_labels[fam_indices]
            fam_probs = test_probs[fam_indices]
            fam_pred = (fam_probs >= threshold).astype(int)

            if fam_labels[0] == 1:
                # Attack family: recall / detection rate is primary
                detected = int(np.sum(fam_pred == 1))
                total = len(fam_labels)
                rate = detected / total if total > 0 else 0.0
                family_results[family] = {
                    "total": total,
                    "detected": detected,
                    "detection_rate": round(rate, 4),
                }
            else:
                # Benign family: false positive count is primary
                fp = int(np.sum(fam_pred == 1))
                total = len(fam_labels)
                fpr = fp / total if total > 0 else 0.0
                family_results[family] = {
                    "total": total,
                    "false_positives": fp,
                    "fpr": round(fpr, 4),
                }

        # 3. Evaluate Hard Negatives Separately
        hard_neg_metrics: Dict[str, Any] = {}
        if self.hard_negatives:
            hn_texts = [ex.text for ex in self.hard_negatives]
            hn_labels = np.zeros(len(hn_texts), dtype=int)
            hn_probs = np.array(predict_fn(hn_texts), dtype=float)

            hn_preds = (hn_probs >= threshold).astype(int)
            hn_fp = int(np.sum(hn_preds == 1))
            hn_total = len(hn_labels)
            hn_fpr = hn_fp / hn_total if hn_total > 0 else 0.0
            hn_ci_fpr = compute_bootstrap_ci(hn_labels, hn_probs, metric_name="fpr", threshold=threshold)

            hard_neg_metrics = {
                "total": hn_total,
                "false_positives": hn_fp,
                "fpr": round(hn_fpr, 4),
                "ci_fpr_95": hn_ci_fpr,
            }

        report = {
            "model_name": model_name,
            "threshold": threshold,
            "test_split_metrics": base_metrics.to_dict(),
            "per_family_metrics": family_results,
            "hard_negative_metrics": hard_neg_metrics,
        }

        return report

    @staticmethod
    def print_report(report: Dict[str, Any]):
        """Prints a clean, formatted evaluation table to console."""
        m_name = report["model_name"]
        thresh = report["threshold"]
        m = report["test_split_metrics"]
        hn = report.get("hard_negative_metrics", {})

        print("\n" + "=" * 72)
        print(f"EVALUATION RESULTS: {m_name} (Threshold: {thresh})")
        print("=" * 72)

        print("\n[HELD-OUT TEST SPLIT]")
        print(f"  Samples:      {m['total_samples']} (TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']})")
        print(f"  Accuracy:     {m['accuracy'] * 100:.2f}%")
        print(f"  Precision:    {m['precision'] * 100:.2f}%")
        print(f"  Recall:       {m['recall'] * 100:.2f}%")
        print(f"  F1 Score:     {m['f1']:.4f}  [95% CI: {m['ci_f1'][0]:.4f} - {m['ci_f1'][1]:.4f}]")
        print(f"  Test FPR:     {m['fpr'] * 100:.2f}%  [95% CI: {m['ci_fpr'][0] * 100:.2f}% - {m['ci_fpr'][1] * 100:.2f}%]")
        print(f"  ROC-AUC:      {m['roc_auc']:.4f}")
        print(f"  PR-AUC:       {m['pr_auc']:.4f}")

        if hn:
            print("\n[HARD NEGATIVES BENCHMARK]")
            print(f"  Total Prompts:       {hn['total']}")
            print(f"  False Positives:     {hn['false_positives']}")
            print(f"  Hard Negatives FPR:  {hn['fpr'] * 100:.2f}%  [95% CI: {hn['ci_fpr_95'][0] * 100:.2f}% - {hn['ci_fpr_95'][1] * 100:.2f}%]")

        print("\n[PER-ATTACK-FAMILY BREAKDOWN (TEST)]")
        fam = report.get("per_family_metrics", {})
        for name, data in fam.items():
            if "detection_rate" in data:
                print(f"  - {name:<18}: {data['detection_rate'] * 100:>6.2f}% detection ({data['detected']}/{data['total']})")
            elif "fpr" in data:
                print(f"  - {name:<18}: {data['fpr'] * 100:>6.2f}% false positives ({data['false_positives']}/{data['total']})")

        print("=" * 72 + "\n")
