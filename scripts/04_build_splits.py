#!/usr/bin/env python3
"""
Step 4: Build train/valid/test JSONL splits with structured responses.
Format: {label} ({confidence}). {reason} Span: "{span}"
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

REMAPPED_DIR = Path(__file__).parent.parent / "data" / "remapped"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
DATA_DIR = Path(__file__).parent.parent / "data"

LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}

OVERCONFIDENCE_SIGNALS = [
    r"\bdefinitively\b", r"\bproven\b", r"\bproves\b", r"\bprove\b",
    r"\balways\b", r"\bnever\b", r"\b100%\b", r"\bconclusively\b",
    r"\bundeniably\b", r"\bfact\b", r"\bscience has shown\b",
    r"\bstudies prove\b", r"\bdefinitely\b", r"\bcertainly\b",
    r"\babsolutely\b", r"\bundoubtedly\b", r"\bwithout a doubt\b",
    r"\bguaranteed?\b", r"\birrefutable\b", r"\bclearly\b", r"\bobviously\b",
    r"\bdefinitive\b", r"\bno doubt\b", r"\bbeyond question\b",
    r"\btruth\b", r"\btrue fact\b",
]

UNDERCONFIDENCE_SIGNALS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b",
    r"\bpossibly\b", r"\bperhaps\b", r"\bmaybe\b",
    r"\blikely\b", r"\bunlikely\b",
    r"\bpotentially\b", r"\bpotential\b",
    r"\bsuggests?\b", r"\bsuggested\b", r"\bsuggesting\b",
    r"\bappears?\b", r"\bseems?\b",
    r"\bindicates?\b", r"\bindicated\b",
    r"\btentative\b", r"\bpreliminary\b",
    r"\bunclear\b", r"\buncertain\b",
    r"\bsome evidence\b", r"\bnot necessarily\b",
    r"\bto some extent\b", r"\btends? to\b",
    r"\blargely\b", r"\bbroadly\b", r"\bgenerally\b",
    r"\brelatively\b", r"\bsomewhat\b",
    r"\bsubject to\b", r"\bdownside risk\b",
    r"\bon balance\b",
]

CONFIDENCE_BY_LABEL_AND_CLARITY = {
    # label: (clarity_high, clarity_medium, clarity_low)
    0: (0.92, 0.78, 0.60),
    1: (0.88, 0.72, 0.55),
    2: (0.95, 0.82, 0.65),
}

REASONS = {
    0: {
        "high": 'The text uses absolute or definitive language that overstates certainty.',
        "neutral": 'The claim expresses confidence that exceeds what the evidence supports.',
        "mixed": 'The phrasing implies more certainty than is warranted by the available evidence.',
    },
    1: {
        "high": 'The text uses excessive hedging where the evidence supports a stronger statement.',
        "neutral": 'Unnecessary qualifiers weaken a claim that could be stated more directly.',
        "mixed": 'The hedging language downplays the strength of the underlying evidence.',
    },
    2: {
        "high": 'The expressed certainty appropriately matches the strength of the evidence.',
        "neutral": 'The language is appropriately calibrated to the level of evidence.',
        "mixed": 'The phrasing reflects reasonable confidence given the evidence.',
    },
}


def find_span(text: str, label: int) -> str:
    """Find the first matching over/under-confidence phrase in text."""
    text_lower = text.lower()
    if label == 0:
        patterns = OVERCONFIDENCE_SIGNALS
    elif label == 1:
        patterns = UNDERCONFIDENCE_SIGNALS
    else:
        return "N/A"
    for p in patterns:
        m = re.search(p, text)
        if m:
            return text[m.start():m.end()]
    if label == 0:
        return "the claim"
    elif label == 1:
        return "hedging language"
    return "N/A"


def compute_confidence(record: dict) -> float:
    """Compute a heuristic confidence score based on label and certainty."""
    label = record["label"]
    certainty = record.get("certainty", "neutral")
    veracity = record.get("veracity_clarity", "clear")

    base_levels = CONFIDENCE_BY_LABEL_AND_CLARITY[label]
    if certainty == "high" and veracity == "clear":
        return base_levels[0]
    elif certainty in ("high", "low") or veracity == "clear":
        return base_levels[1]
    else:
        return base_levels[2]


def detect_clarity(record: dict) -> str:
    """Determine veracity clarity from original label."""
    orig = record.get("original_label", "").lower()
    if orig in ("contradict", "support", "false", "true", "refuted", "supported",
                 "refutes", "supports", "certain"):
        return "clear"
    return "unclear"


def to_chat_format(record: dict) -> dict:
    """Convert a record to chat format with structured response."""
    text = record["text"]
    label = record["label"]
    label_name = LABEL_NAMES[label]

    span = find_span(text, label)
    clarity = detect_clarity(record)
    record["veracity_clarity"] = clarity
    confidence = compute_confidence(record)
    certainty = record.get("certainty", "neutral")

    if certainty == "high" and label == 0:
        reason_key = "high"
    elif certainty == "low" and label == 1:
        reason_key = "high"
    elif certainty == "low" and label == 2:
        reason_key = "mixed"
    elif certainty == "high" and label == 2:
        reason_key = "mixed"
    else:
        reason_key = "neutral"
    reason = REASONS[label][reason_key]

    if span and span != "N/A":
        response = f"{label_name} ({confidence:.2f}). {reason} Span: \"{span}\""
    else:
        response = f"{label_name} ({confidence:.2f}). {reason}"

    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response},
        ]
    }


def load_all_records() -> list[dict]:
    records = []
    for f in sorted(REMAPPED_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                records.append(json.loads(line))
    syn_path = SYNTHETIC_DIR / "synthetic.jsonl"
    if syn_path.exists():
        with open(syn_path) as fh:
            for line in fh:
                records.append(json.loads(line))
    return records


def balance_classes(records: list[dict], max_ratio: float = 1.5) -> list[dict]:
    if not records:
        return []
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    if len(by_label) < 3:
        print("    [WARN] One or more calibration classes are missing. Skipping balancing.")
        return records
    min_count = min(len(v) for v in by_label.values())
    if min_count == 0:
        print("    [WARN] Minority class has 0 records. Skipping balancing.")
        return records
    cap = int(min_count * max_ratio)
    balanced = []
    for label, items in by_label.items():
        if len(items) > cap:
            random.shuffle(items)
            balanced.extend(items[:cap])
        else:
            balanced.extend(items)
    random.shuffle(balanced)
    return balanced


HEDGING_PREFIXES = [
    "Some researchers suggest that ",
    "It is possible that ",
    "There is some evidence that ",
    "Preliminary findings indicate that ",
    "It may be the case that ",
    "The data tentatively suggests that ",
    "Early analysis indicates that ",
    "One interpretation is that ",
    "To some extent, ",
    "It could be argued that ",
]

OVERCONFIDENCE_PREFIXES = [
    "Studies definitively prove that ",
    "It is unquestionably true that ",
    "Science has clearly shown that ",
    "There is no doubt that ",
    "It is an absolute fact that ",
    "Research conclusively demonstrates that ",
    "It is irrefutable that ",
    "Experts unanimously agree that ",
    "The evidence clearly indicates that ",
    "It is well-established that ",
]


def generate_synthetic_uc(record: dict) -> dict:
    """Generate an UNDERCLAIMING example by adding hedging to a CALIBRATED sentence."""
    text = record["text"]
    prefix = random.choice(HEDGING_PREFIXES)
    # Avoid double-hedging if text already starts with hedging
    if any(text.lower().startswith(p.lower().rstrip(" that ").rstrip()) for p in HEDGING_PREFIXES):
        return None
    new_text = prefix + text[0].lower() + text[1:]
    return {
        "text": new_text,
        "label": 1,
        "source": f"synthetic_uc_{record.get('source', 'unknown')}",
        "certainty": "low",
        "original_label": "hedged_calibrated",
        "veracity_clarity": "unclear",
    }


def generate_synthetic_oc(record: dict) -> dict:
    """Generate an OVERCLAIMING example by adding overconfidence to a CALIBRATED sentence."""
    text = record["text"]
    prefix = random.choice(OVERCONFIDENCE_PREFIXES)
    new_text = prefix + text[0].lower() + text[1:]
    return {
        "text": new_text,
        "label": 0,
        "source": f"synthetic_oc_{record.get('source', 'unknown')}",
        "certainty": "high",
        "original_label": "overconfident_calibrated",
        "veracity_clarity": "clear",
    }


def main():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 4: Building splits with structured responses")
    print("=" * 60)

    all_records = load_all_records()
    print(f"  Total records loaded: {len(all_records)}")

    # Rule-based synthetic augmentation for class balance
    cal_records = [r for r in all_records if r["label"] == 2]
    oc_records = [r for r in all_records if r["label"] == 0]
    uc_records = [r for r in all_records if r["label"] == 1]

    needed_uc = min(max(0, len(oc_records) - len(uc_records)), 10000)
    needed_oc_extra = min(max(0, len(uc_records) - len(oc_records)), 5000)

    if needed_uc > 0:
        random.shuffle(cal_records)
        uc_synthetic = []
        for r in cal_records[:needed_uc * 2]:
            syn = generate_synthetic_uc(r)
            if syn:
                uc_synthetic.append(syn)
                if len(uc_synthetic) >= needed_uc:
                    break
        all_records.extend(uc_synthetic)
        print(f"  Generated {len(uc_synthetic)} synthetic UNDERCLAIMING examples")

    if needed_oc_extra > 0:
        random.shuffle(cal_records)
        oc_synthetic = []
        for r in cal_records[:needed_oc_extra * 2]:
            syn = generate_synthetic_oc(r)
            if syn:
                oc_synthetic.append(syn)
                if len(oc_synthetic) >= needed_oc_extra:
                    break
        all_records.extend(oc_synthetic)
        print(f"  Generated {len(oc_synthetic)} synthetic OVERCLAIMING examples")

    # Multi-source held-out: 70/30 split per source
    by_source = defaultdict(list)
    for r in all_records:
        by_source[r.get("source", "unknown")].append(r)

    train_records = []
    test_records = []
    for source, records in by_source.items():
        random.shuffle(records)
        split = int(len(records) * 0.7)
        train_records.extend(records[:split])
        test_records.extend(records[split:])

    print(f"  Multi-source train pool: {len(train_records)}")
    print(f"  Multi-source test pool: {len(test_records)}")

    # Balance training set
    balanced = balance_classes(train_records)
    print(f"  After balancing: {len(balanced)}")

    split_idx = int(len(balanced) * 0.9)
    train_final = balanced[:split_idx]
    valid_final = balanced[split_idx:]

    # Subsample and balance test set to ~1500 samples for practical evaluation
    test_balanced = balance_classes(test_records)
    # Cap each class at 500 for test (max 1500 total)
    by_label_test = defaultdict(list)
    for r in test_balanced:
        by_label_test[r["label"]].append(r)
    test_final = []
    for label, items in by_label_test.items():
        random.shuffle(items)
        test_final.extend(items[:500])
    random.shuffle(test_final)

    print(f"  Test set after subsampling: {len(test_final)}")
    dist_test = defaultdict(int)
    for r in test_final:
        dist_test[r["label"]] += 1
    for label in sorted(dist_test):
        print(f"    {LABEL_NAMES[label]}: {dist_test[label]} ({100*dist_test[label]/len(test_final):.1f}%)")

    splits = {"train": train_final, "valid": valid_final, "test": test_final}

    for split_name, records in splits.items():
        out_path = DATA_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                chat = to_chat_format(r)
                f.write(json.dumps(chat) + "\n")

        dist = defaultdict(int)
        for r in records:
            dist[r["label"]] += 1

        print(f"\n  {split_name}: {len(records)} records")
        for label in sorted(dist):
            print(f"    {LABEL_NAMES[label]}: {dist[label]} ({100*dist[label]/len(records):.1f}%)")
        print(f"    → {out_path}")

    # Show samples
    print(f"\n{'=' * 60}")
    print("Sample outputs:")
    for split_name in ["train", "valid", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        with open(path) as f:
            samples = [json.loads(line) for line in f][:2]
        print(f"\n  {split_name} samples:")
        for s in samples:
            user = s["messages"][0]["content"]
            asst = s["messages"][1]["content"]
            print(f"    User: {user[:80]}...")
            print(f"    Asst: {asst[:120]}")
            print()

    print(f"{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
