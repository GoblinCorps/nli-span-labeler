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

    # Sample mode - random stratified samples for quick bootstrapping
    python scripts/download_data.py --sample
    python scripts/download_data.py --sample 500
    python scripts/download_data.py --sample 500 --import
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed.")
    print("Install with: pip install datasets")
    exit(1)


LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "nli"
DB_PATH = Path(__file__).parent.parent / "labels.db"  # Must match app.py DB_PATH
DEFAULT_SAMPLE_SIZE = 500


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


def stratified_sample(dataset, split: str, sample_size: int) -> list:
    """
    Stratified random sampling to maintain label distribution.

    Returns approximately sample_size examples with balanced labels.
    """
    # Group examples by label
    by_label = defaultdict(list)
    for idx, ex in enumerate(dataset[split]):
        label = ex.get("label", -1)
        if label != -1:  # Skip unlabeled
            by_label[label].append((idx, ex))

    # Calculate per-label count (divide evenly)
    per_label = sample_size // len(by_label)
    remainder = sample_size % len(by_label)

    sampled = []
    for i, (label, examples) in enumerate(sorted(by_label.items())):
        # Distribute remainder to first labels
        count = per_label + (1 if i < remainder else 0)
        count = min(count, len(examples))  # Don't exceed available

        # Random sample
        selected = random.sample(examples, count)
        sampled.extend(selected)

    return sampled


def sample_snli(sample_size: int) -> list:
    """Download and sample SNLI dataset."""
    print("Downloading SNLI for sampling...")
    dataset = load_dataset("stanfordnlp/snli")

    all_samples = []
    # Sample from train split (largest)
    samples = stratified_sample(dataset, "train", sample_size)
    for idx, ex in samples:
        record = convert_example(ex, "snli", "train", idx)
        if record:
            all_samples.append(record)

    print(f"  SNLI: {len(all_samples)} samples")
    return all_samples


def sample_mnli(sample_size: int) -> list:
    """Download and sample MNLI dataset."""
    print("Downloading MNLI for sampling...")
    dataset = load_dataset("nyu-mll/multi_nli")

    all_samples = []
    # Sample from train split
    samples = stratified_sample(dataset, "train", sample_size)
    for idx, ex in samples:
        record = convert_example(ex, "mnli", "train", idx)
        if record:
            all_samples.append(record)

    print(f"  MNLI: {len(all_samples)} samples")
    return all_samples


def sample_anli(sample_size: int) -> list:
    """Download and sample ANLI dataset."""
    print("Downloading ANLI for sampling...")
    dataset = load_dataset("facebook/anli")

    all_samples = []
    # Sample evenly from all rounds (r1, r2, r3)
    per_round = sample_size // 3
    for round_name in ["train_r1", "train_r2", "train_r3"]:
        samples = stratified_sample(dataset, round_name, per_round)
        for idx, ex in samples:
            record = convert_example(ex, "anli", round_name, idx)
            if record:
                all_samples.append(record)

    print(f"  ANLI: {len(all_samples)} samples")
    return all_samples


def import_to_database(examples: list, db_path: Path):
    """Import examples directly into the annotation database."""
    print(f"\nImporting {len(examples)} examples to {db_path}...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if examples table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='examples'"
    )
    if not cursor.fetchone():
        print("Error: Database not initialized. Run the app first to create tables.")
        conn.close()
        return False

    imported = 0
    skipped = 0
    for ex in examples:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO examples (id, premise, hypothesis, label, dataset_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ex["id"], ex["premise"], ex["hypothesis"], ex["label_text"], ex["dataset_name"])
            )
            if conn.total_changes > imported + skipped:
                imported += 1
            else:
                skipped += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    conn.close()

    print(f"  Imported: {imported}")
    print(f"  Skipped (already exist): {skipped}")
    return True


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


def run_sample_mode(sample_size: int, datasets: list, do_import: bool):
    """Run sample mode - download stratified random samples."""
    print(f"Sample mode: ~{sample_size} examples per dataset")
    print(f"Datasets: {', '.join(datasets)}")
    print()

    all_samples = []

    if "snli" in datasets:
        all_samples.extend(sample_snli(sample_size))

    if "mnli" in datasets:
        all_samples.extend(sample_mnli(sample_size))

    if "anli" in datasets:
        all_samples.extend(sample_anli(sample_size))

    # Save to JSONL file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sample.jsonl"
    with open(output_path, "w") as f:
        for record in all_samples:
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal: {len(all_samples)} samples -> {output_path}")

    # Label distribution
    label_counts = defaultdict(int)
    for ex in all_samples:
        label_counts[ex["label_text"]] += 1
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/len(all_samples):.1f}%)")

    # Import to database if requested
    if do_import:
        import_to_database(all_samples, DB_PATH)

    return all_samples


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
    parser.add_argument(
        "--sample",
        nargs="?",
        const=DEFAULT_SAMPLE_SIZE,
        type=int,
        metavar="N",
        help=f"Sample mode: download ~N random examples per dataset (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="Import samples directly into annotation database (requires --sample)",
    )
    args = parser.parse_args()

    # Sample mode
    if args.sample is not None:
        run_sample_mode(args.sample, args.datasets, args.do_import)
        print("\nDone!")
        return

    # Full download mode (original behavior)
    if args.do_import:
        print("Error: --import requires --sample mode")
        exit(1)

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
