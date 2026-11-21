"""
Dataset downloader for public prompt injection and jailbreak datasets.

Downloads:
1. deepset/prompt-injections (Apache-2.0)
2. jackhhao/jailbreak-classification (MIT / Research)

Saves raw JSON payloads into data/raw/ directory.
Uses direct Hugging Face Datasets API (requires only requests, no auth token).
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import requests

DATASETS = {
    "deepset": {
        "repo": "deepset/prompt-injections",
        "splits": ["train", "test"],
        "license": "Apache-2.0",
    },
    "jackhhao": {
        "repo": "jackhhao/jailbreak-classification",
        "splits": ["train", "test"],
        "license": "MIT",
    },
}

API_BASE = "https://datasets-server.huggingface.co/rows"


def fetch_dataset_split(repo: str, split: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetches all rows for a given dataset split via the Hugging Face server API."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    total = None

    print(f"--> Fetching {repo} [{split}]...")

    while True:
        url = f"{API_BASE}?dataset={requests.utils.quote(repo)}&config=default&split={split}&offset={offset}&limit={page_size}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error fetching offset {offset} for {repo} [{split}]: {e}")
            break

        if total is None:
            total = data.get("num_rows_total", 0)

        current_rows = [item["row"] for item in data.get("rows", [])]
        if not current_rows:
            break

        rows.extend(current_rows)
        offset += len(current_rows)
        print(f"    Progress: {offset}/{total} rows downloaded")

        if offset >= total:
            break

        time.sleep(0.1)  # polite rate-limiting

    print(f"--> Completed {repo} [{split}]: {len(rows)} rows.")
    return rows


def download_all(output_dir: str = "data/raw") -> Dict[str, str]:
    """Downloads all defined datasets into the raw data directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for name, config in DATASETS.items():
        repo = config["repo"]
        for split in config["splits"]:
            rows = fetch_dataset_split(repo, split)
            file_name = f"{name}_{split}.json"
            target_file = out_path / file_name

            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)

            saved_files[f"{name}_{split}"] = str(target_file)
            print(f"Saved: {target_file}")

    return saved_files


if __name__ == "__main__":
    download_all()
