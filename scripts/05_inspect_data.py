#!/usr/bin/env python3
"""
Step 5: Inspect the final dataset — print stats, class distribution, sample entries.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}


def inspect_split(path: Path):
    """Print stats for a single JSONL split file."""
    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"\n  {path.name}: {len(records)} records")

    # Extract labels from chat format
    label_counts = defaultdict(int)
    lengths = []
    samples = {0: None, 1: None, 2: None}

    for r in records:
        msgs = r.get("messages", [])
        if len(msgs) >= 2:
            user_text = msgs[0]["content"]
            assistant_text = msgs[1]["content"]
            lengths.append(len(user_text))

            for label_id, name in LABEL_NAMES.items():
                if assistant_text.startswith(name):
                    label_counts[label_id] += 1
                    if samples[label_id] is None:
                        samples[label_id] = user_text.replace(
                            "Classify the confidence calibration of this text:\n\n", ""
                        )
                    break

    # Distribution
    total = sum(label_counts.values())
    print(f"    Class distribution:")
    for label_id in sorted(label_counts):
        count = label_counts[label_id]
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"      {LABEL_NAMES[label_id]:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Length stats
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"    Avg input length: {avg_len:.0f} chars")
        print(f"    Min/Max length: {min(lengths)}/{max(lengths)} chars")

    # Sample entries
    print(f"    Sample entries:")
    for label_id, sample in samples.items():
        if sample:
            truncated = sample[:100] + "..." if len(sample) > 100 else sample
            print(f"      [{LABEL_NAMES[label_id]}] {truncated}")


def check_leakage(train_path: Path, valid_path: Path, test_path: Path):
    """Check for data leakage between splits."""
    def load_texts(path):
        texts = set()
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("messages"):
                        texts.add(r["messages"][0]["content"])
        return texts

    train = load_texts(train_path)
    valid = load_texts(valid_path)
    test = load_texts(test_path)

    tv = train & valid
    tt = train & test
    vt = valid & test

    print(f"\n  Leakage check:")
    print(f"    Train ∩ Valid: {len(tv)} overlaps {'⚠️' if tv else '✅'}")
    print(f"    Train ∩ Test:  {len(tt)} overlaps {'⚠️' if tt else '✅'}")
    print(f"    Valid ∩ Test:  {len(vt)} overlaps {'⚠️' if vt else '✅'}")


def main():
    print("=" * 60)
    print("STEP 5: Dataset inspection")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        inspect_split(DATA_DIR / f"{split}.jsonl")

    check_leakage(
        DATA_DIR / "train.jsonl",
        DATA_DIR / "valid.jsonl",
        DATA_DIR / "test.jsonl",
    )

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
