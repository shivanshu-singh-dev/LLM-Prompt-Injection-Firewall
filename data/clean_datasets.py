"""
Cleans and standardizes raw dataset files into the PromptExample schema.
"""

from pathlib import Path
from typing import List, Optional
import json
from data.schema import PromptExample
from data.convert_scanner import load_scanner_payloads


def clean_deepset(file_path: Path, split_name: str) -> List[PromptExample]:
    """Cleans deepset/prompt-injections raw JSON file."""
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    examples = []
    for idx, item in enumerate(raw_data):
        text = item.get("text", "").strip()
        if not text:
            continue
        raw_label = int(item.get("label", 0))
        label = 1 if raw_label == 1 else 0
        family = "prompt_injection" if label == 1 else "benign"

        examples.append(
            PromptExample(
                id=f"deepset_{split_name}_{idx:04d}",
                text=text,
                label=label,
                attack_family=family,
                source="deepset/prompt-injections",
                split="unassigned",
                metadata={
                    "original_split": split_name,
                    "license": "Apache-2.0",
                },
            )
        )
    return examples


def clean_jackhhao(file_path: Path, split_name: str) -> List[PromptExample]:
    """Cleans jackhhao/jailbreak-classification raw JSON file."""
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    examples = []
    for idx, item in enumerate(raw_data):
        text = item.get("prompt", "").strip()
        if not text:
            continue
        item_type = str(item.get("type", "")).strip().lower()
        label = 1 if item_type == "jailbreak" else 0
        family = "jailbreak" if label == 1 else "benign"

        examples.append(
            PromptExample(
                id=f"jackhhao_{split_name}_{idx:04d}",
                text=text,
                label=label,
                attack_family=family,
                source="jackhhao/jailbreak-classification",
                split="unassigned",
                metadata={
                    "original_split": split_name,
                    "license": "MIT",
                },
            )
        )
    return examples


def load_all_datasets(raw_dir: str = "data/raw", payloads_path: str = "payloads/payloads.yaml") -> List[PromptExample]:
    """
    Loads and normalizes:
    - 69 scanner payloads
    - deepset/prompt-injections (if present in raw_dir)
    - jackhhao/jailbreak-classification (if present in raw_dir)
    """
    all_examples: List[PromptExample] = []

    # 1. 69 Scanner payloads
    scanner_examples = load_scanner_payloads(payloads_path)
    all_examples.extend(scanner_examples)
    print(f"Loaded {len(scanner_examples)} scanner payloads.")

    raw_path = Path(raw_dir)
    if raw_path.exists():
        # 2. Deepset
        for split in ["train", "test"]:
            deepset_file = raw_path / f"deepset_{split}.json"
            if deepset_file.exists():
                cleaned = clean_deepset(deepset_file, split)
                all_examples.extend(cleaned)
                print(f"Loaded {len(cleaned)} examples from {deepset_file.name}")

        # 3. Jackhhao
        for split in ["train", "test"]:
            jack_file = raw_path / f"jackhhao_{split}.json"
            if jack_file.exists():
                cleaned = clean_jackhhao(jack_file, split)
                all_examples.extend(cleaned)
                print(f"Loaded {len(cleaned)} examples from {jack_file.name}")

    print(f"Total raw examples aggregated: {len(all_examples)}")
    return all_examples


if __name__ == "__main__":
    examples = load_all_datasets()
