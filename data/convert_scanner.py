"""
Converts payloads/payloads.yaml (the 69 scanner adversarial payloads)
into the standardized PromptExample JSONL format.
"""

from pathlib import Path
from typing import List
import yaml
from data.schema import PromptExample


def load_scanner_payloads(payloads_path: str = "payloads/payloads.yaml") -> List[PromptExample]:
    """Reads scanner payloads.yaml and converts to a list of PromptExample objects."""
    path = Path(payloads_path)
    if not path.exists():
        raise FileNotFoundError(f"Payloads file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_payloads = yaml.safe_load(f)

    examples: List[PromptExample] = []
    for item in raw_payloads:
        example_id = f"scanner_{item['id']}"
        category = item.get("category", "prompt_injection")
        technique = item.get("technique", "")
        prompt_text = item.get("prompt", "").strip()
        expected_marker = item.get("expected_marker", "NW-7734-ADMIN")

        if not prompt_text:
            continue

        examples.append(
            PromptExample(
                id=example_id,
                text=prompt_text,
                label=1,  # Malicious attack
                attack_family=category,
                source="scanner_payloads",
                split="unassigned",
                metadata={
                    "technique": technique,
                    "expected_marker": expected_marker,
                    "original_id": item["id"],
                },
            )
        )

    return examples


if __name__ == "__main__":
    payloads = load_scanner_payloads()
    print(f"Successfully converted {len(payloads)} scanner payloads into PromptExample format.")
    for p in payloads[:3]:
        print(f"[{p.attack_family}] {p.id}: {p.text[:60]}...")
