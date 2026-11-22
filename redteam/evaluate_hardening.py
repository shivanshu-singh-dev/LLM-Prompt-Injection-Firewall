"""
Evaluator for Adversarial Hardening (Phase 7).

Benchmarks Bypass Rates before and after adversarial retraining on a strictly
held-out test set of mutated attacks that were NEVER used during retraining.
Also verifies False Positive Rate on hard negatives to ensure specificity is preserved.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np

from redteam.attack_firewall import load_seed_payloads, generate_mutated_attack_suite
from redteam.adversarial_retraining import split_mutations_train_test, train_hardened_classifier
from firewall.engine import FirewallEngine
from eval.metrics import compute_binary_metrics


def evaluate_suite_bypass_rate(
    engine: FirewallEngine,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates bypass rate of a firewall engine across a list of attack items."""
    bypassed_count = 0
    per_type: Dict[str, Dict[str, int]] = {}

    for item in items:
        mut_type = item["mutation_type"]
        prompt = item["mutated_prompt"]

        if mut_type not in per_type:
            per_type[mut_type] = {"total": 0, "bypassed": 0}
        per_type[mut_type]["total"] += 1

        inspection = engine.inspect_prompt(prompt, session_id=f"bench-{item['payload_id']}")
        if inspection["verdict"] != "BLOCK":
            bypassed_count += 1
            per_type[mut_type]["bypassed"] += 1

    total = len(items)
    overall_rate = (bypassed_count / total * 100.0) if total > 0 else 0.0

    breakdown = {}
    for m_type, d in per_type.items():
        rate = (d["bypassed"] / d["total"] * 100.0) if d["total"] > 0 else 0.0
        breakdown[m_type] = {
            "total": d["total"],
            "bypassed": d["bypassed"],
            "bypass_rate_pct": round(rate, 2),
        }

    return {
        "total_tested": total,
        "total_bypassed": bypassed_count,
        "bypass_rate_pct": round(overall_rate, 2),
        "per_type_breakdown": breakdown,
    }


def run_hardening_benchmark():
    print("=" * 80)
    print("PROMPT INJECTION FIREWALL — ADAPTIVE RED-TEAMING & HARDENING (PHASE 7)")
    print("=" * 80)

    # 1. Load seed payloads and generate mutated attack suite
    print("--> Generating mutated adversarial attack suite from 69 scanner payloads...")
    seed = load_seed_payloads()
    all_mutations = generate_mutated_attack_suite(seed)
    print(f"    Generated {len(all_mutations)} adversarial test vectors across 6 mutation categories.")

    # 2. Split mutations: 60% retraining, 40% strictly held-out test set
    train_muts, heldout_test_muts = split_mutations_train_test(all_mutations, train_ratio=0.60, seed=42)
    print(f"    Partitioned into:")
    print(f"      - Retraining Augmentation Set: {len(train_muts)} attacks")
    print(f"      - Strictly Held-Out Test Set:  {len(heldout_test_muts)} attacks (never seen by retraining)")

    # 3. Benchmark BEFORE Retraining on Held-Out Test Set
    print("\n--> [Step 1/3] Benchmarking Initial Firewall on Held-Out Mutations...")
    engine_initial = FirewallEngine()
    results_before = evaluate_suite_bypass_rate(engine_initial, heldout_test_muts)
    print(f"    Initial Bypass Rate on Held-Out Set: {results_before['bypass_rate_pct']:.2f}% ({results_before['total_bypassed']}/{results_before['total_tested']})")

    # 4. Perform Adversarial Retraining
    print("\n--> [Step 2/3] Executing Adversarial Retraining Pipeline...")
    hardened_model = train_hardened_classifier(
        clean_train_path="data/processed/train.jsonl",
        adversarial_train_items=train_muts,
        output_path="models/hardened/hardened_classifier.joblib",
    )

    # 5. Benchmark AFTER Retraining on Held-Out Test Set
    print("\n--> [Step 3/3] Benchmarking Hardened Firewall on Held-Out Mutations...")
    engine_hardened = FirewallEngine()
    # Plug hardened model into firewall engine
    engine_hardened.model = hardened_model
    engine_hardened.model_name = "Hardened Classifier (Adversarial)"

    results_after = evaluate_suite_bypass_rate(engine_hardened, heldout_test_muts)
    print(f"    Hardened Bypass Rate on Held-Out Set: {results_after['bypass_rate_pct']:.2f}% ({results_after['total_bypassed']}/{results_after['total_tested']})")

    # 6. Check Specificity on Hard Negatives
    print("\n--> Checking Specificity on Hard Negatives...")
    hn_path = Path("data/processed/hard_negatives.jsonl")
    hn_fpr_before = 0.0
    hn_fpr_after = 0.0
    if hn_path.exists():
        with open(hn_path, "r", encoding="utf-8") as f:
            hn_texts = [json.loads(line)["text"] for line in f if line.strip()]

        hn_blocked_before = sum(1 for t in hn_texts if engine_initial.inspect_prompt(t)["verdict"] == "BLOCK")
        hn_blocked_after = sum(1 for t in hn_texts if engine_hardened.inspect_prompt(t)["verdict"] == "BLOCK")
        hn_fpr_before = (hn_blocked_before / len(hn_texts)) * 100.0
        hn_fpr_after = (hn_blocked_after / len(hn_texts)) * 100.0
        print(f"    Hard Negatives Block Rate: Before = {hn_fpr_before:.2f}% | After = {hn_fpr_after:.2f}%")

    # 7. Print Comparative Table
    print("\n" + "=" * 80)
    print("ADVERSARIAL HARDENING RESULTS (EVALUATED ON HELD-OUT MUTATION TEST SET)")
    print("=" * 80)
    print(f"{'Mutation Category':<22} | {'Before Bypass Rate':<20} | {'After Bypass Rate':<20} | {'Bypass Reduction':<16}")
    print("-" * 80)

    for m_type in sorted(results_before["per_type_breakdown"].keys()):
        b_rate = results_before["per_type_breakdown"][m_type]["bypass_rate_pct"]
        a_rate = results_after["per_type_breakdown"][m_type]["bypass_rate_pct"]
        reduction = b_rate - a_rate
        print(f"{m_type:<22} | {b_rate:>6.2f}%              | {a_rate:>6.2f}%              | -{reduction:>5.2f}%")

    print("-" * 80)
    tot_before = results_before["bypass_rate_pct"]
    tot_after = results_after["bypass_rate_pct"]
    tot_reduction = tot_before - tot_after
    print(f"{'OVERALL HELD-OUT':<22} | {tot_before:>6.2f}%              | {tot_after:>6.2f}%              | -{tot_reduction:>5.2f}%")
    print("=" * 80)

    reduction_pct = ((tot_before - tot_after) / tot_before * 100.0) if tot_before > 0 else 0.0
    print(f"\n--> Successfully reduced overall evasion bypass rate by {reduction_pct:.1f}% relative reduction on unseen test mutations!\n")


if __name__ == "__main__":
    run_hardening_benchmark()
