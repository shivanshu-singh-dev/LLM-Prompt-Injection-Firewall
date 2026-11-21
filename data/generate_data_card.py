"""
Generates DATA_CARD.md documenting dataset provenance, counts, splits, and licenses.
"""

from collections import Counter
from pathlib import Path
from typing import List, Dict
from data.schema import PromptExample


def generate_data_card(
    splits: Dict[str, List[PromptExample]],
    exact_dups_removed: int = 0,
    near_dups_removed: int = 0,
    output_path: str = "DATA_CARD.md",
) -> str:
    """Generates a comprehensive markdown data card for the dataset."""
    all_examples = []
    for split_examples in splits.values():
        all_examples.extend(split_examples)

    total_count = len(all_examples)
    sources_counter = Counter(ex.source for ex in all_examples)
    families_counter = Counter(ex.attack_family for ex in all_examples)
    labels_counter = Counter(ex.label for ex in all_examples)

    # Split breakdown
    split_counts = {name: len(items) for name, items in splits.items()}

    content = f"""# Data Card: Prompt Injection Firewall Dataset

**Date Generated:** 2026-11-21  
**Project:** LLM Prompt Injection Firewall  
**Total Examples:** {total_count}  

---

## 1. Overview & Purpose
This dataset is curated specifically to train and evaluate production prompt injection firewalls. It combines red-team payloads from the original scanner test suite with verified public injection and jailbreak datasets, balanced by benign queries and hard negatives.

---

## 2. Dataset Sources & Licenses

| Source | Count | Proportion | License | Description |
|---|---|---|---|---|
| `scanner_payloads` | {sources_counter.get('scanner_payloads', 0)} | {sources_counter.get('scanner_payloads', 0) / total_count * 100:.1f}% | MIT | Original 69 adversarial payloads targeting system prompt extraction & instruction override |
| `deepset/prompt-injections` | {sources_counter.get('deepset/prompt-injections', 0)} | {sources_counter.get('deepset/prompt-injections', 0) / total_count * 100:.1f}% | Apache-2.0 | Curated public prompt injections and benign controls |
| `jackhhao/jailbreak-classification` | {sources_counter.get('jackhhao/jailbreak-classification', 0)} | {sources_counter.get('jackhhao/jailbreak-classification', 0) / total_count * 100:.1f}% | MIT | Community jailbreak prompts and conversational benign requests |

---

## 3. Split Distribution (80 / 10 / 10 Stratified)

All splits are stratified across `(label, attack_family)` pairs with random seed `42`.

| Split | Count | Benign (0) | Malicious (1) | % of Total |
|---|---|---|---|---|
| **Train** | {split_counts.get('train', 0)} | {sum(1 for x in splits.get('train', []) if x.label == 0)} | {sum(1 for x in splits.get('train', []) if x.label == 1)} | {split_counts.get('train', 0) / total_count * 100:.1f}% |
| **Validation** | {split_counts.get('val', 0)} | {sum(1 for x in splits.get('val', []) if x.label == 0)} | {sum(1 for x in splits.get('val', []) if x.label == 1)} | {split_counts.get('val', 0) / total_count * 100:.1f}% |
| **Test** | {split_counts.get('test', 0)} | {sum(1 for x in splits.get('test', []) if x.label == 0)} | {sum(1 for x in splits.get('test', []) if x.label == 1)} | {split_counts.get('test', 0) / total_count * 100:.1f}% |
| **Total** | **{total_count}** | **{labels_counter.get(0, 0)}** | **{labels_counter.get(1, 0)}** | **100.0%** |

> [!IMPORTANT]
> The **Test** split is strictly held out. It must never be used during model training, hyperparameter tuning, or threshold calibration.

---

## 4. Attack Families & Taxonomy

Every example is assigned an `attack_family` to support Leave-One-Family-Out (LOFO) generalization benchmarking:

| Attack Family | Count | Primary Focus |
|---|---|---|
| `benign` | {families_counter.get('benign', 0)} | Legitimate user queries, conversational prompts, and benign helpdesk questions |
| `role_override` | {families_counter.get('role_override', 0)} | Policy puppetry (XML/JSON), system role spoofing, authority framing, credential claims |
| `jailbreak` | {families_counter.get('jailbreak', 0)} | Persona switching (DAN/AIM), hypothetical framing, emotional coercion |
| `indirect` | {families_counter.get('indirect', 0)} | Embedded malicious instructions in secondary data (logs, tickets, emails) |
| `obedience_exploit` | {families_counter.get('obedience_exploit', 0)} | Syntactic obedience traps (verbatim repetition, grammar correction, completion) |
| `prompt_injection` | {families_counter.get('prompt_injection', 0)} | Direct prompt injection patterns from public datasets |

---

## 5. Deduplication & Data Quality

To prevent data leakage across splits and avoid artificial accuracy inflation:
- **Exact matches removed:** {exact_dups_removed}
- **Near-duplicates removed:** {near_dups_removed} (evaluated via character 3-gram Jaccard similarity ≥ 0.88)
- Scanner payloads are given highest retention priority during deduplication.
"""
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    print(f"Data card generated at: {out_file}")
    return content


if __name__ == "__main__":
    from data.schema import PromptExample

    dummy_splits = {
        "train": [PromptExample("1", "test prompt", 1, "role_override", "scanner_payloads")],
        "val": [],
        "test": [],
    }
    generate_data_card(dummy_splits)
