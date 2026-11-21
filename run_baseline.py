"""
Single-command reproducible entrypoint for Baseline Classifier (Phase 2).

Trains TF-IDF + Logistic Regression baseline, tunes threshold on validation data,
evaluates on held-out test split, benchmarks against hard negatives, measures
per-sample inference latency, and prints the headline benchmark metrics.

Usage:
    python run_baseline.py
"""

import json
import time
from pathlib import Path
import numpy as np
from data.schema import PromptExample
from data.curate_hard_negatives import export_hard_negatives
from models.baseline_tfidf import BaselineTFIDFClassifier
from eval.evaluator import FirewallEvaluator


def load_split(split_file: str) -> list[PromptExample]:
    """Loads a list of PromptExample objects from a JSONL file."""
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Required split file {path} not found. Ensure Phase 1 dataset is built via 'python -m data.build_dataset'."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [PromptExample.from_json(line) for line in f if line.strip()]


def measure_latency(model: BaselineTFIDFClassifier, sample_texts: list[str], runs: int = 200) -> tuple[float, float]:
    """Measures single-request inference latency (p50 and p95 in milliseconds)."""
    # Warmup
    for _ in range(10):
        _ = model.predict_proba([sample_texts[0]])

    latencies_ms = []
    # Test batch_size=1 sequentially to simulate real-time proxy inference
    for i in range(runs):
        text = sample_texts[i % len(sample_texts)]
        t0 = time.perf_counter()
        _ = model.predict_proba([text])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return round(p50, 2), round(p95, 2)


def main():
    print("=" * 72)
    print("PROMPT INJECTION FIREWALL — BASELINE EVALUATION (PHASE 2)")
    print("=" * 72)

    # 1. Ensure hard negatives exist
    hard_neg_path = Path("data/processed/hard_negatives.jsonl")
    if not hard_neg_path.exists():
        print("--> Generating hard negatives dataset...")
        export_hard_negatives(str(hard_neg_path))

    # 2. Load splits
    print("--> Loading dataset splits...")
    train_data = load_split("data/processed/train.jsonl")
    val_data = load_split("data/processed/val.jsonl")
    test_data = load_split("data/processed/test.jsonl")
    hard_neg_data = load_split("data/processed/hard_negatives.jsonl")

    print(f"    Train samples:          {len(train_data)}")
    print(f"    Validation samples:     {len(val_data)}")
    print(f"    Held-out Test samples:  {len(test_data)}")
    print(f"    Hard Negatives samples: {len(hard_neg_data)}")

    # 3. Train Baseline Model
    print("\n--> Fitting TF-IDF + Logistic Regression pipeline...")
    train_texts = [ex.text for ex in train_data]
    train_labels = [ex.label for ex in train_data]

    model = BaselineTFIDFClassifier(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        c_param=1.0,
        random_state=42,
    )
    t_start = time.perf_counter()
    model.fit(train_texts, train_labels)
    t_train = time.perf_counter() - t_start
    print(f"    Training completed in {t_train:.2f}s.")

    # 4. Tune decision threshold on Validation split
    val_texts = [ex.text for ex in val_data]
    val_labels = [ex.label for ex in val_data]

    calibrated_threshold = model.tune_threshold_on_validation(
        val_texts=val_texts,
        val_labels=val_labels,
        target_fpr=0.02,
        strategy="target_fpr",
    )
    print(f"    Calibrated decision threshold for validation FPR <= 2%: {calibrated_threshold:.3f}")

    # 5. Save model artifact
    saved_path = model.save("models/baseline/baseline_tfidf.joblib")
    print(f"    Saved baseline model to {saved_path}")

    # 6. Evaluate on Test Split and Hard Negatives
    evaluator = FirewallEvaluator(test_examples=test_data, hard_negatives=hard_neg_data)

    print("\n--> Evaluating at default threshold (0.50)...")
    report_default = evaluator.evaluate(
        predict_fn=model.predict_proba,
        threshold=0.50,
        model_name="TF-IDF + LogReg (Default 0.50)",
    )
    evaluator.print_report(report_default)

    print("--> Evaluating at calibrated threshold...")
    report_calibrated = evaluator.evaluate(
        predict_fn=model.predict_proba,
        threshold=calibrated_threshold,
        model_name=f"TF-IDF + LogReg (Calibrated {calibrated_threshold:.3f})",
    )
    evaluator.print_report(report_calibrated)

    # 7. Measure Single-Request Latency
    test_texts = [ex.text for ex in test_data]
    p50, p95 = measure_latency(model, test_texts)

    # 8. Headline Summary Table
    m_cal = report_calibrated["test_split_metrics"]
    hn_cal = report_calibrated["hard_negative_metrics"]

    print("\n" + "=" * 72)
    print("HEADLINE BENCHMARK TABLE")
    print("=" * 72)
    print(f"{'Model':<25} | {'Test F1':<12} | {'Test FPR':<10} | {'Hard Neg FPR':<14} | {'p95 Latency':<12}")
    print("-" * 80)
    print(
        f"{'TF-IDF Baseline (Calibrated)':<25} | "
        f"{m_cal['f1']:.4f}       | "
        f"{m_cal['fpr'] * 100:.2f}%     | "
        f"{hn_cal['fpr'] * 100:.2f}%        | "
        f"{p95:.2f} ms"
    )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
