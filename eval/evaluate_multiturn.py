"""
Multi-Turn Conversational Defense Evaluator (Phase 6).

Compares:
1. Single-Message Detection: scores each turn in complete isolation.
2. Multi-Turn Defense: evaluates sliding window concatenation and exponential risk accumulation.

Measures:
- Adversarial detection rate on multi-turn crescendo and split attacks.
- False positive rate on multi-turn benign dialogues.
- Conversational F1 improvement.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

from firewall.engine import FirewallEngine
from firewall.multi_turn import MultiTurnSessionTracker
from eval.metrics import compute_binary_metrics


def load_multiturn_conversations(path: str = "data/processed/multiturn_conversations.jsonl") -> List[Dict[str, Any]]:
    """Loads multi-turn conversations from JSONL."""
    p = Path(path)
    if not p.exists():
        from data.curate_multiturn import export_multiturn_dataset
        export_multiturn_dataset(str(p))

    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_multiturn_defense(
    conversations: List[Dict[str, Any]],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates Single-Turn vs Multi-Turn defense across all conversations.
    """
    engine = FirewallEngine(threshold_flag=0.35, threshold_block=threshold)

    y_true = []
    single_turn_detected = []
    multiturn_detected = []

    for convo in conversations:
        convo_id = convo["id"]
        true_label = int(convo["label"])
        y_true.append(true_label)

        turns = convo["turns"]
        session_tracker = MultiTurnSessionTracker(window_size=3, decay_factor=0.65)

        convo_single_flagged = False
        convo_multi_flagged = False

        for turn in turns:
            if turn["role"] == "user":
                user_text = turn["content"]

                # 1. Single-message evaluation in isolation
                single_score = float(engine.model.predict_proba([user_text])[0])
                rule_hit, _, _ = engine.rule_engine.inspect(user_text)
                if single_score >= threshold or rule_hit:
                    convo_single_flagged = True

                # 2. Multi-turn sliding window + risk accumulator
                window_text = session_tracker.get_sliding_window_text(convo_id, user_text)
                window_score = float(engine.model.predict_proba([window_text])[0])

                cum_risk = session_tracker.record_turn_and_accumulate_risk(
                    session_id=convo_id,
                    user_text=user_text,
                    turn_score=single_score,
                    window_score=window_score,
                )
                if rule_hit:
                    cum_risk = max(cum_risk, 0.75)

                if cum_risk >= threshold:
                    convo_multi_flagged = True

            elif turn["role"] == "assistant":
                session_tracker.record_assistant_response(convo_id, turn["content"])

        single_turn_detected.append(1 if convo_single_flagged else 0)
        multiturn_detected.append(1 if convo_multi_flagged else 0)

    y_true_arr = np.array(y_true, dtype=int)
    single_arr = np.array(single_turn_detected, dtype=int)
    multi_arr = np.array(multiturn_detected, dtype=int)

    # Compute metrics for Single-Turn baseline
    single_metrics = compute_binary_metrics(y_true_arr, single_arr.astype(float), threshold=0.5)
    # Compute metrics for Multi-Turn defense
    multi_metrics = compute_binary_metrics(y_true_arr, multi_arr.astype(float), threshold=0.5)

    adv_indices = np.where(y_true_arr == 1)[0]
    benign_indices = np.where(y_true_arr == 0)[0]

    single_adv_recall = float(np.mean(single_arr[adv_indices])) if len(adv_indices) > 0 else 0.0
    multi_adv_recall = float(np.mean(multi_arr[adv_indices])) if len(adv_indices) > 0 else 0.0

    single_benign_fpr = float(np.mean(single_arr[benign_indices])) if len(benign_indices) > 0 else 0.0
    multi_benign_fpr = float(np.mean(multi_arr[benign_indices])) if len(benign_indices) > 0 else 0.0

    report = {
        "total_conversations": len(conversations),
        "adversarial_conversations": int(len(adv_indices)),
        "benign_conversations": int(len(benign_indices)),
        "single_turn": {
            "detection_rate": round(single_adv_recall * 100, 2),
            "false_positive_rate": round(single_benign_fpr * 100, 2),
            "f1": single_metrics.f1,
            "precision": round(single_metrics.precision * 100, 2),
        },
        "multi_turn": {
            "detection_rate": round(multi_adv_recall * 100, 2),
            "false_positive_rate": round(multi_benign_fpr * 100, 2),
            "f1": multi_metrics.f1,
            "precision": round(multi_metrics.precision * 100, 2),
        },
        "detection_gain_pct": round((multi_adv_recall - single_adv_recall) * 100, 2),
        "fpr_delta_pct": round((multi_benign_fpr - single_benign_fpr) * 100, 2),
    }

    return report


def print_multiturn_report(report: Dict[str, Any]):
    """Prints a clean summary comparing single-turn vs multi-turn defense."""
    print("\n" + "=" * 76)
    print("MULTI-TURN CONVERSATIONAL DEFENSE BENCHMARK (PHASE 6)")
    print("=" * 76)
    print(f"Total Dialogues Tested:        {report['total_conversations']}")
    print(f"Adversarial Trajectories:     {report['adversarial_conversations']}")
    print(f"Benign Dialogues:             {report['benign_conversations']}")
    print("-" * 76)
    print(f"{'Defense Architecture':<24} | {'Attack Detection':<18} | {'Benign FPR':<14} | {'F1 Score':<10}")
    print("-" * 76)

    st = report["single_turn"]
    mt = report["multi_turn"]
    print(f"{'Single-Turn Isolation':<24} | {st['detection_rate']:>6.2f}%            | {st['false_positive_rate']:>5.2f}%        | {st['f1']:.4f}")
    print(f"{'Multi-Turn Window+Risk':<24} | {mt['detection_rate']:>6.2f}%            | {mt['false_positive_rate']:>5.2f}%        | {mt['f1']:.4f}")
    print("-" * 76)
    print(f"Detection Improvement:        +{report['detection_gain_pct']:.2f}%")
    print(f"FPR Change:                   {report['fpr_delta_pct']:+.2f}%")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    convos = load_multiturn_conversations()
    rep = evaluate_multiturn_defense(convos)
    print_multiturn_report(rep)
