"""
Stratified Train/Val/Test splitting pipeline.

Splits data into 80% train, 10% val, 10% test while stratifying
by (label, attack_family) pairs to ensure identical distribution across splits.
Uses fixed random seed for strict reproducibility.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
from data.schema import PromptExample


def stratified_split(
    examples: List[PromptExample],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[PromptExample]]:
    """
    Performs stratified train/val/test splitting grouped by (label, attack_family).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group by stratum: (label, attack_family)
    strata: Dict[Tuple[int, str], List[PromptExample]] = defaultdict(list)
    for ex in examples:
        strata[(ex.label, ex.attack_family)].append(ex)

    splits: Dict[str, List[PromptExample]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for stratum_key, items in strata.items():
        # Shuffle deterministically within stratum
        items_copy = list(items)
        rng.shuffle(items_copy)

        n = len(items_copy)
        if n == 1:
            # Special case for singleton items: place in train
            items_copy[0].split = "train"
            splits["train"].append(items_copy[0])
            continue

        n_val = max(1, int(round(n * val_ratio))) if n >= 5 else 1
        n_test = max(1, int(round(n * test_ratio))) if n >= 5 else 1
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            n_val = max(0, n - 2)
            n_test = 1

        val_items = items_copy[:n_val]
        test_items = items_copy[n_val:n_val + n_test]
        train_items = items_copy[n_val + n_test:]

        for item in train_items:
            item.split = "train"
            splits["train"].append(item)

        for item in val_items:
            item.split = "val"
            splits["val"].append(item)

        for item in test_items:
            item.split = "test"
            splits["test"].append(item)

    return splits


def export_splits(
    splits: Dict[str, List[PromptExample]],
    output_dir: str = "data/processed",
    master_filename: str = "dataset.jsonl",
) -> Dict[str, Path]:
    """Exports train, val, test, and master JSONL files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_paths = {}

    # 1. Master file containing all examples
    master_path = out_dir / master_filename
    with open(master_path, "w", encoding="utf-8") as f_master:
        for split_name in ["train", "val", "test"]:
            for ex in splits[split_name]:
                f_master.write(ex.to_json() + "\n")
    file_paths["master"] = master_path

    # 2. Individual split files
    for split_name in ["train", "val", "test"]:
        split_path = out_dir / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f_split:
            for ex in splits[split_name]:
                f_split.write(ex.to_json() + "\n")
        file_paths[split_name] = split_path

    return file_paths
