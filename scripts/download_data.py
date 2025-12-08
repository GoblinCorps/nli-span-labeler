#!/usr/bin/env python3
"""
Download and convert NLI datasets for the span labeler.

Downloads from Hugging Face:
- SNLI (Stanford NLI) - ~570K examples
- MNLI (Multi-Genre NLI) - ~433K examples
- ANLI (Adversarial NLI) - ~170K examples

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --datasets snli mnli
    python scripts/download_data.py --limit 1000  # For testing
"""

import argparse
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed.")
    print("Install with: pip install datasets")
    exit(1)


LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "nli"


def convert_example(ex: dict, dataset_name: str, split: str, idx: int) -> dict:
    """Convert a Hugging Face example to our format."""
    label = ex.get("label", -1)
    if label == -1:  # Skip examples without gold labels
        return None

    return {
        "id": f"{dataset_name}_{split}_{idx}",
        "premise": ex["premise"],
        "hypothesis": ex["hypothesis"],
        "label": label,
        "label_text": LABEL_MAP.get(label, "unknown"),
        "dataset_name": dataset_name,
        "split": split,
    }


def download_snli(limit: int = None):
    """Download and convert SNLI dataset."""
    print("Downloading SNLI...")
    dataset = load_dataset("stanfordnlp/snli")

    for split in ["train", "validation", "test"]:
        output_path = OUTPUT_DIR / f"snli_{split}.jsonl"
        count = 0

        with open(output_path, "w") as f:
            for idx, ex in enumerate(dataset[split]):
                if limit and count >= limit:
                    break
                record = convert_example(ex, "snli", split, idx)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1

        print(f"  {split}: {count} examples -> {output_path}")


def download_mnli(limit: int = None):
    """Download and convert MNLI dataset."""
    print("Downloading MNLI...")
    dataset = load_dataset("nyu-mll/multi_nli")

    split_map = {
        "train": "train",
        "validation_matched": "validation_matched",
        "validation_mismatched": "validation_mismatched",
    }

    for hf_split, our_split in split_map.items():
        output_path = OUTPUT_DIR / f"mnli_{our_split}.jsonl"
        count = 0

        with open(output_path, "w") as f:
            for idx, ex in enumerate(dataset[hf_split]):
                if limit and count >= limit:
                    break
                record = convert_example(ex, "mnli", our_split, idx)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1

        print(f"  {our_split}: {count} examples -> {output_path}")


def download_anli(limit: int = None):
    """Download and convert ANLI dataset."""
    print("Downloading ANLI...")
    dataset = load_dataset("facebook/anli")

    for split in ["train_r1", "train_r2", "train_r3", "dev_r1", "dev_r2", "dev_r3", "test_r1", "test_r2", "test_r3"]:
        output_path = OUTPUT_DIR / f"anli_{split}.jsonl"
        count = 0

        with open(output_path, "w") as f:
            for idx, ex in enumerate(dataset[split]):
                if limit and count >= limit:
                    break
                record = convert_example(ex, "anli", split, idx)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1

        print(f"  {split}: {count} examples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download NLI datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["snli", "mnli", "anli"],
        choices=["snli", "mnli", "anli"],
        help="Which datasets to download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per split (for testing)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if "snli" in args.datasets:
        download_snli(args.limit)
        print()

    if "mnli" in args.datasets:
        download_mnli(args.limit)
        print()

    if "anli" in args.datasets:
        download_anli(args.limit)
        print()

    print("Done!")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
