"""
Leave-One-Attack-Family-Out (LOFO) Generalization Benchmark.

Evaluates how well a model generalizes to zero-day / unseen attack families.
Iteratively holds out one complete attack family from the training set,
trains on the remaining families, and evaluates detection rate and F1 on the held-out family.

The gap between random split F1 and LOFO F1 is the empirical Generalization Drop.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from data.schema import PromptExample
from models.baseline_tfidf import BaselineTFIDFClassifier
from eval.metrics import compute_binary_metrics


CORE_ATTACK_FAMILIES = [
    "role_override",
    "jailbreak",
    "indirect",
    "obedience_exploit",
]


def load_all_dataset_examples(dataset_file: str = "data/processed/dataset.jsonl") -> List[PromptExample]:
    """Loads all examples from the master dataset."""
    path = Path(dataset_file)
    with open(path, "r", encoding="utf-8") as f:
        return [PromptExample.from_json(line) for line in f if line.strip()]


def run_lofo_benchmark(
    all_examples: List[PromptExample],
    families: List[str] = CORE_ATTACK_FAMILIES,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Executes Leave-One-Attack-Family-Out cross-validation.
    
    Returns:
        Dictionary containing per-family LOFO metrics, macro-average LOFO F1,
        and comparison against full random split.
    """
    # Baseline on full random split for comparison
    train_full = [ex for ex in all_examples if ex.split == "train"]
    test_full = [ex for ex in all_examples if ex.split == "test"]

    full_model = BaselineTFIDFClassifier()
    full_model.fit([ex.text for ex in train_full], [ex.label for ex in train_full])
    full_test_probs = full_model.predict_proba([ex.text for ex in test_full])
    full_test_labels = [ex.label for ex in test_full]
    full_metrics = compute_binary_metrics(np.array(full_test_labels), full_test_probs, threshold=threshold)

    lofo_results: Dict[str, Any] = {}
    family_drops: Dict[str, float] = {}

    for held_out_family in families:
        # 1. Training set: Exclude all examples belonging to the held-out attack family
        train_lofo = [
            ex for ex in all_examples
            if ex.split == "train" and ex.attack_family != held_out_family
        ]

        # 2. Test set: Held-out family examples + benign test controls
        test_lofo = [
            ex for ex in all_examples
            if ex.split == "test" and (ex.attack_family == held_out_family or ex.label == 0)
        ]

        if not any(ex.attack_family == held_out_family for ex in test_lofo):
            continue

        # 3. Train model strictly without held-out family
        lofo_model = BaselineTFIDFClassifier()
        lofo_model.fit([ex.text for ex in train_lofo], [ex.label for ex in train_lofo])

        # 4. Evaluate on held-out family test partition
        y_test = np.array([ex.label for ex in test_lofo], dtype=int)
        y_probs = lofo_model.predict_proba([ex.text for ex in test_lofo])
        metrics = compute_binary_metrics(y_test, y_probs, threshold=threshold)

        # Calculate family detection rate
        family_indices = [i for i, ex in enumerate(test_lofo) if ex.attack_family == held_out_family]
        family_preds = (y_probs[family_indices] >= threshold).astype(int)
        detected = int(np.sum(family_preds == 1))
        total_family = len(family_indices)
        lofo_detection_rate = detected / total_family if total_family > 0 else 0.0

        # Calculate detection rate with full model (random split)
        full_family_indices = [i for i, ex in enumerate(test_full) if ex.attack_family == held_out_family]
        full_family_preds = (full_test_probs[full_family_indices] >= threshold).astype(int)
        full_detected = int(np.sum(full_family_preds == 1))
        full_detection_rate = full_detected / len(full_family_indices) if full_family_indices else 0.0

        drop = full_detection_rate - lofo_detection_rate
        family_drops[held_out_family] = drop

        lofo_results[held_out_family] = {
            "samples_in_test": total_family,
            "lofo_detection_rate": round(lofo_detection_rate * 100, 2),
            "random_split_detection_rate": round(full_detection_rate * 100, 2),
            "detection_drop_pct": round(drop * 100, 2),
            "lofo_f1": metrics.f1,
            "lofo_fpr": round(metrics.fpr * 100, 2),
        }

    macro_lofo_f1 = float(np.mean([res["lofo_f1"] for res in lofo_results.values()]))
    macro_drop = float(np.mean(list(family_drops.values())))

    summary = {
        "random_split_f1": full_metrics.f1,
        "macro_lofo_f1": round(macro_lofo_f1, 4),
        "f1_generalization_gap": round(full_metrics.f1 - macro_lofo_f1, 4),
        "mean_family_detection_drop_pct": round(macro_drop * 100, 2),
        "per_family_lofo": lofo_results,
    }

    return summary


def print_lofo_report(summary: Dict[str, Any]):
    """Prints a clean LOFO generalization report table."""
    print("\n" + "=" * 76)
    print("LEAVE-ONE-ATTACK-FAMILY-OUT (LOFO) GENERALIZATION BENCHMARK")
    print("=" * 76)
    print(f"Random Split Test F1:       {summary['random_split_f1']:.4f}")
    print(f"Macro LOFO F1 (Unseen):     {summary['macro_lofo_f1']:.4f}")
    print(f"Generalization F1 Gap:      {summary['f1_generalization_gap']:.4f}")
    print(f"Mean Detection Rate Drop:   {summary['mean_family_detection_drop_pct']:.2f}%")
    print("-" * 76)
    print(f"{'Held-Out Family':<20} | {'Random Split':<14} | {'LOFO (Unseen)':<14} | {'Accuracy Drop':<14}")
    print("-" * 76)

    for fam, d in summary["per_family_lofo"].items():
        print(
            f"{fam:<20} | "
            f"{d['random_split_detection_rate']:>6.2f}%        | "
            f"{d['lofo_detection_rate']:>6.2f}%        | "
            f"-{d['detection_drop_pct']:>5.2f}%"
        )
    print("=" * 76 + "\n")


if __name__ == "__main__":
    dataset = load_all_dataset_examples()
    summary = run_lofo_benchmark(dataset)
    print_lofo_report(summary)
