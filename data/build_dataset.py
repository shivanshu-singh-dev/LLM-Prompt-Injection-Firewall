"""
Master Dataset Builder for Prompt Injection Firewall.

Executes the complete Phase 1 pipeline:
1. Ingests 69 scanner payloads from payloads/payloads.yaml
2. Ingests public datasets (deepset, jackhhao) from data/raw/ (or triggers download if missing)
3. Deduplicates exact and near-duplicate prompts
4. Performs stratified 80/10/10 train/val/test splitting
5. Exports data/processed/dataset.jsonl, train.jsonl, val.jsonl, test.jsonl
6. Generates DATA_CARD.md
"""

import argparse
from pathlib import Path
from data.convert_scanner import load_scanner_payloads
from data.clean_datasets import load_all_datasets
from data.deduplicate import deduplicate_examples
from data.split import stratified_split, export_splits
from data.generate_data_card import generate_data_card
from data.download_datasets import download_all


def build_firewall_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    payloads_path: str = "payloads/payloads.yaml",
    download_if_missing: bool = True,
    near_dup_threshold: float = 0.88,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
):
    print("=" * 60)
    print("PROMPT INJECTION FIREWALL - DATASET BUILDER (PHASE 1)")
    print("=" * 60)

    raw_path = Path(raw_dir)
    raw_files = list(raw_path.glob("*.json")) if raw_path.exists() else []

    # Step 1: Check / Download public datasets
    if (not raw_files) and download_if_missing:
        print("[Step 1] Raw datasets not found in data/raw. Initiating download...")
        download_all(raw_dir)
    else:
        print(f"[Step 1] Found {len(raw_files)} raw dataset files in {raw_dir}.")

    # Step 2: Load and standardize all sources
    print("\n[Step 2] Ingesting and normalizing all dataset sources...")
    all_examples = load_all_datasets(raw_dir=raw_dir, payloads_path=payloads_path)
    total_raw = len(all_examples)
    print(f"Total raw examples collected: {total_raw}")

    # Step 3: Deduplicate
    print(f"\n[Step 3] Running deduplication (near_dup_threshold={near_dup_threshold})...")
    unique_examples, exact_dups, near_dups = deduplicate_examples(
        all_examples, near_dup_threshold=near_dup_threshold
    )
    print(f"Deduplication complete:")
    print(f"  - Exact duplicates removed: {exact_dups}")
    print(f"  - Near-duplicates removed: {near_dups}")
    print(f"  - Unique examples retained: {len(unique_examples)}")

    # Step 4: Stratified Train/Val/Test Split
    print(f"\n[Step 4] Stratified splitting ({int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)})...")
    splits = stratified_split(
        unique_examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for name, items in splits.items():
        benign_count = sum(1 for x in items if x.label == 0)
        malicious_count = sum(1 for x in items if x.label == 1)
        print(f"  - {name.upper()}: {len(items)} examples ({benign_count} benign, {malicious_count} malicious)")

    # Step 5: Export JSONL files
    print(f"\n[Step 5] Exporting JSONL files to {processed_dir}...")
    file_paths = export_splits(splits, output_dir=processed_dir, master_filename="dataset.jsonl")
    for key, path in file_paths.items():
        print(f"  - Wrote {key}: {path} ({path.stat().st_size / 1024:.1f} KB)")

    # Step 6: Generate Data Card
    print("\n[Step 6] Generating DATA_CARD.md...")
    generate_data_card(
        splits=splits,
        exact_dups_removed=exact_dups,
        near_dups_removed=near_dups,
        output_path="DATA_CARD.md",
    )

    print("\n[SUCCESS] Phase 1 dataset build completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Prompt Injection Firewall dataset.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading if raw data is missing")
    parser.add_argument("--near-dup-threshold", type=float, default=0.88, help="Near-duplicate threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    build_firewall_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download_if_missing=not args.no_download,
        near_dup_threshold=args.near_dup_threshold,
        seed=args.seed,
    )
