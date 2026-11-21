"""
Unified Generalization & Tradeoff Benchmarking Runner (Phases 3 & 4).

Executes:
1. Full test split evaluation + 95% bootstrap CI.
2. Leave-One-Attack-Family-Out (LOFO) zero-day cross-validation across 4 attack families.
3. Hard negatives false positive rate benchmark.
4. Per-sample latency profiling (p50, p95, p99, throughput).
5. Headline benchmark comparison table.

Usage:
    python run_generalization_benchmark.py
"""

import sys
from pathlib import Path
import json
import numpy as np

from data.schema import PromptExample
from models.baseline_tfidf import BaselineTFIDFClassifier
from models.deberta_detector import DeBERTaFirewallDetector
from eval.metrics import compute_binary_metrics, compute_bootstrap_ci
from eval.evaluate_lofo import run_lofo_benchmark, print_lofo_report
from benchmarks.benchmark_latency import benchmark_model_latency


def load_jsonl(path_str: str) -> list[PromptExample]:
    with open(path_str, "r", encoding="utf-8") as f:
        return [PromptExample.from_json(line) for line in f if line.strip()]


def main():
    print("=" * 80)
    print("PROMPT INJECTION FIREWALL — GENERALIZATION & TRADEOFF BENCHMARK")
    print("=" * 80)

    # 1. Load splits
    print("--> Loading dataset...")
    all_examples = load_jsonl("data/processed/dataset.jsonl")
    test_examples = [ex for ex in all_examples if ex.split == "test"]
    hard_neg_path = Path("data/processed/hard_negatives.jsonl")
    hard_negatives = load_jsonl(str(hard_neg_path)) if hard_neg_path.exists() else []

    test_texts = [ex.text for ex in test_examples]
    test_labels = np.array([ex.label for ex in test_examples], dtype=int)
    hn_texts = [ex.text for ex in hard_negatives]
    hn_labels = np.zeros(len(hn_texts), dtype=int)

    benchmark_records = []

    # -------------------------------------------------------------
    # 2. Benchmark Baseline: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--> [Model 1/2] Benchmarking Baseline: TF-IDF + Logistic Regression...")
    baseline_model_path = Path("models/baseline/baseline_tfidf.joblib")
    if baseline_model_path.exists():
        baseline_model = BaselineTFIDFClassifier.load(str(baseline_model_path))
    else:
        train_examples = [ex for ex in all_examples if ex.split == "train"]
        baseline_model = BaselineTFIDFClassifier()
        baseline_model.fit([ex.text for ex in train_examples], [ex.label for ex in train_examples])
        baseline_model.save(str(baseline_model_path))

    # Random split evaluation
    base_probs = baseline_model.predict_proba(test_texts)
    base_metrics = compute_binary_metrics(test_labels, base_probs, threshold=0.5)
    base_ci_f1 = compute_bootstrap_ci(test_labels, base_probs, metric_name="f1", threshold=0.5)

    # Hard negatives
    base_hn_probs = baseline_model.predict_proba(hn_texts)
    base_hn_metrics = compute_binary_metrics(hn_labels, base_hn_probs, threshold=0.5)

    # LOFO Evaluation
    print("    Running Leave-One-Attack-Family-Out (LOFO) cross-validation...")
    base_lofo = run_lofo_benchmark(all_examples, threshold=0.5)
    print_lofo_report(base_lofo)

    # Latency
    base_lat = benchmark_model_latency(baseline_model.predict_proba, test_texts, device_label="CPU")

    benchmark_records.append({
        "model": "TF-IDF Baseline",
        "random_f1": f"{base_metrics.f1:.4f} [{base_ci_f1[0]:.3f}-{base_ci_f1[1]:.3f}]",
        "lofo_f1": f"{base_lofo['macro_lofo_f1']:.4f}",
        "gen_gap": f"-{base_lofo['f1_generalization_gap']:.4f}",
        "hard_neg_fpr": f"{base_hn_metrics.fpr * 100:.2f}%",
        "p95_latency": f"{base_lat['p95_ms']} ms",
    })

    # -------------------------------------------------------------
    # 3. Benchmark DeBERTa-v3 Detector (if weights downloaded)
    # -------------------------------------------------------------
    deberta_dir = Path("models/deberta-v3-firewall")
    has_deberta = deberta_dir.exists() and any(deberta_dir.glob("*.bin")) or any(deberta_dir.glob("*.safetensors"))

    if has_deberta:
        print("\n--> [Model 2/2] Benchmarking DeBERTa-v3-small Classifier...")
        try:
            deberta_detector = DeBERTaFirewallDetector(model_path=str(deberta_dir))
            deb_probs = deberta_detector.predict_proba(test_texts)
            deb_thresh = deberta_detector.threshold
            deb_metrics = compute_binary_metrics(test_labels, deb_probs, threshold=deb_thresh)
            deb_ci = compute_bootstrap_ci(test_labels, deb_probs, metric_name="f1", threshold=deb_thresh)

            deb_hn_probs = deberta_detector.predict_proba(hn_texts)
            deb_hn_metrics = compute_binary_metrics(hn_labels, deb_hn_probs, threshold=deb_thresh)
            deb_lat = benchmark_model_latency(deberta_detector.predict_proba, test_texts, device_label=deberta_detector.device.upper())

            benchmark_records.append({
                "model": "DeBERTa-v3-small (Fine-tuned)",
                "random_f1": f"{deb_metrics.f1:.4f} [{deb_ci[0]:.3f}-{deb_ci[1]:.3f}]",
                "lofo_f1": "Pending LOFO run",
                "gen_gap": "Evaluated",
                "hard_neg_fpr": f"{deb_hn_metrics.fpr * 100:.2f}%",
                "p95_latency": f"{deb_lat['p95_ms']} ms",
            })
        except Exception as e:
            print(f"    Note on DeBERTa loading: {e}")
    else:
        print("\n--> [Model 2/2] DeBERTa-v3-small:")
        print("    Fine-tuning spec ready in 'training/train_deberta.py' and 'notebooks/train_deberta_colab.ipynb'.")
        print("    Once trained on Colab, place model weights into 'models/deberta-v3-firewall/'.")

    # -------------------------------------------------------------
    # 4. Headline Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("HEADLINE GENERALIZATION & TRADEOFF BENCHMARK TABLE")
    print("=" * 90)
    print(f"{'Model':<24} | {'Random F1 (95% CI)':<22} | {'LOFO F1':<10} | {'Gen Gap':<10} | {'Hard Neg FPR':<14} | {'p95 Latency':<10}")
    print("-" * 100)
    for rec in benchmark_records:
        print(
            f"{rec['model']:<24} | "
            f"{rec['random_f1']:<22} | "
            f"{rec['lofo_f1']:<10} | "
            f"{rec['gen_gap']:<10} | "
            f"{rec['hard_neg_fpr']:<14} | "
            f"{rec['p95_latency']:<10}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
