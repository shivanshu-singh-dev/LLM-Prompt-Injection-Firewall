"""
Automated Red-Teaming Attack Runner against Prompt Injection Firewall.

Takes seed payloads from payloads/payloads.yaml, applies adversarial mutations
(Base64, ROT13, Leetspeak, Code-block, Multilingual, Payload splitting),
attacks the firewall, logs bypasses (False Negatives), and computes bypass rates.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import yaml

from redteam.mutators import AdversarialMutator
from firewall.engine import FirewallEngine


def load_seed_payloads(payloads_path: str = "payloads/payloads.yaml") -> List[Dict[str, Any]]:
    """Loads seed payloads from YAML."""
    path = Path(payloads_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed payloads not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def generate_mutated_attack_suite(
    payloads: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generates mutated payloads for all seed attacks."""
    suite = []
    for item in payloads:
        p_id = item["id"]
        category = item.get("category", "unknown")
        raw_prompt = item.get("prompt", "").strip()
        if not raw_prompt:
            continue

        mutations = AdversarialMutator.generate_all_mutations(raw_prompt, p_id)
        for m in mutations:
            m["category"] = category
            suite.append(m)

    return suite


def attack_firewall_suite(
    suite: Optional[List[Dict[str, Any]]] = None,
    engine: Optional[FirewallEngine] = None,
    output_log: str = "results/redteam_bypasses.jsonl",
) -> Dict[str, Any]:
    """
    Attacks the firewall with the mutated payload suite.
    
    Returns:
        Summary dictionary containing total attacks, bypass counts,
        overall bypass rate, and per-mutation breakdown.
    """
    if engine is None:
        engine = FirewallEngine()

    if suite is None:
        seed = load_seed_payloads()
        suite = generate_mutated_attack_suite(seed)

    out_path = Path(output_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bypasses = []
    category_stats: Dict[str, Dict[str, int]] = {}

    with open(out_path, "w", encoding="utf-8") as f_out:
        for item in suite:
            mut_type = item["mutation_type"]
            prompt = item["mutated_prompt"]
            p_id = item["payload_id"]

            if mut_type not in category_stats:
                category_stats[mut_type] = {"total": 0, "bypassed": 0, "blocked": 0}
            category_stats[mut_type]["total"] += 1

            # Inspect prompt with firewall
            inspection = engine.inspect_prompt(prompt, session_id=f"redteam-{p_id}")
            verdict = inspection["verdict"]
            risk_score = inspection["risk_score"]

            is_bypassed = (verdict != "BLOCK")  # ALLOW or FLAG means it bypassed the block filter

            record = {
                "payload_id": p_id,
                "base_id": item["base_id"],
                "category": item.get("category"),
                "mutation_type": mut_type,
                "verdict": verdict,
                "risk_score": risk_score,
                "rule_matched": inspection.get("rule_matched"),
                "bypassed": is_bypassed,
                "prompt": prompt,
            }

            if is_bypassed:
                bypasses.append(record)
                category_stats[mut_type]["bypassed"] += 1
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                category_stats[mut_type]["blocked"] += 1

    total_attacks = len(suite)
    total_bypasses = len(bypasses)
    bypass_rate = (total_bypasses / total_attacks * 100.0) if total_attacks > 0 else 0.0

    breakdown = {}
    for mut_type, stats in category_stats.items():
        rate = (stats["bypassed"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        breakdown[mut_type] = {
            "total": stats["total"],
            "bypassed": stats["bypassed"],
            "blocked": stats["blocked"],
            "bypass_rate_pct": round(rate, 2),
        }

    summary = {
        "total_attacks": total_attacks,
        "total_bypassed": total_bypasses,
        "total_blocked": total_attacks - total_bypasses,
        "bypass_rate_pct": round(bypass_rate, 2),
        "per_mutation_breakdown": breakdown,
        "bypasses_log_file": str(out_path),
    }

    return summary


def print_attack_report(summary: Dict[str, Any]):
    """Prints a formatted attack summary report."""
    print("\n" + "=" * 76)
    print("ADAPTIVE RED-TEAMING ATTACK REPORT")
    print("=" * 76)
    print(f"Total Mutated Attacks:         {summary['total_attacks']}")
    print(f"Attacks Blocked by Firewall:   {summary['total_blocked']}")
    print(f"Attacks Bypassed (Escaped):    {summary['total_bypassed']}")
    print(f"OVERALL BYPASS RATE:           {summary['bypass_rate_pct']:.2f}%")
    print("-" * 76)
    print(f"{'Mutation Strategy':<22} | {'Total Tested':<14} | {'Bypassed':<12} | {'Bypass Rate':<12}")
    print("-" * 76)

    for m_type, d in summary["per_mutation_breakdown"].items():
        print(f"{m_type:<22} | {d['total']:<14} | {d['bypassed']:<12} | {d['bypass_rate_pct']:>6.2f}%")

    print("=" * 76)
    print(f"Bypasses saved to: {summary['bypasses_log_file']}\n")


if __name__ == "__main__":
    rep = attack_firewall_suite()
    print_attack_report(rep)
