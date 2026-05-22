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

    # Separate synthetic and real records — preserve all synthetic
    synthetic_prefixes = ("synthetic_",)
    for label, items in by_label.items():
        synthetic = [r for r in items if r.get("source", "").startswith(synthetic_prefixes)]
        real = [r for r in items if not r.get("source", "").startswith(synthetic_prefixes)]
        by_label[label] = {"synthetic": synthetic, "real": real}

    min_real = min(len(v["real"]) + len(v["synthetic"]) for v in by_label.values())
    if min_real == 0:
        print("    [WARN] Minority class has 0 records. Skipping balancing.")
        return [r for v in by_label.values() for r in v["real"] + v["synthetic"]]

    min_after_keep = min(len(v["synthetic"]) + int(len(v["real"]) * 0.3) for v in by_label.values())
    cap = max(int(min_after_keep * max_ratio), min_real)

    balanced = []
    for label in sorted(by_label.keys()):
        syn = by_label[label]["synthetic"]
        real = by_label[label]["real"]
        # Keep all synthetic
        keep = list(syn)
        # Fill remaining cap with real data
        remaining = cap - len(syn)
        if remaining > 0 and real:
            random.shuffle(real)
            keep.extend(real[:remaining])
        elif remaining <= 0:
            random.shuffle(syn)
            keep = syn[:cap]
        balanced.extend(keep)
    random.shuffle(balanced)
    return balanced


HEDGING_PREFIXES = [
    "Some researchers suggest that ",
    "It is possible that ",
    "There is some evidence that ",
    "Preliminary findings indicate that ",
    "It may be the case that ",
    "The data tentatively suggests that ",
    "One interpretation is that ",
    "To some extent, ",
    "It could be argued that ",
    "The results may indicate that ",
]

# Subtle overconfidence patterns — natural-sounding, not blatant trigger phrases
OVERCONFIDENCE_AMPLIFIERS = [
    (r"\b(may|might|possibly)\b", "will"),               # "may cause" → "will cause"
    (r"\bsuggests?\b", "proves"),                          # "suggests" → "proves"
    (r"\bindicates?\b", "demonstrates"),                   # "indicates" → "demonstrates"
    (r"\bappears? to\b", ""),                              # "appears to" → remove
    (r"\bseems? to\b", ""),                                # "seems to" → remove
    (r"\bcould\b", "does"),                                # "could cause" → "does cause"
    (r"\bpotentially\b", ""),                              # "potentially" → remove
    (r"\bsome evidence\b", "strong evidence"),             # "some evidence" → "strong evidence"
    (r"\bin some cases\b", "in all cases"),                # generalize
    (r"\boften\b", "always"),                              # generalize to absolute
    (r"\bmaybe\b", "certainly"),                           # hedge → certainty
    (r"\bperhaps\b", "undoubtedly"),                       # hedge → certainty
    (r"\btends? to\b", "invariably"),                      # "tends to" → "invariably"
    (r"\blargely\b", "completely"),                        # amplify
]

OVERCONFIDENCE_INSERTIONS = [
    "Clearly, ", "Undoubtedly, ", "Of course, ",
    "It is clear that ", "There can be no doubt that ",
    "The truth is that ", "As we know, ",
]

# Evidence qualifiers to turn OC claims into CAL (hard negatives)
EVIDENCE_PREFIXES = [
    "According to a 2023 study, ",
    "Research published in a peer-reviewed journal indicates that ",
    "A recent meta-analysis found that ",
    "Clinical trial data suggests that ",
    "Observational studies have shown that ",
    "The available evidence indicates that ",
    "In a controlled experiment, researchers found that ",
    "Epidemiological data suggests a link between ",
    "According to preliminary findings, ",
    "A systematic review concluded that ",
]


def apply_overconfidence(text: str) -> str:
    """Apply subtle overconfidence transformations to a sentence."""
    for pattern, replacement in OVERCONFIDENCE_AMPLIFIERS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Clean up double spaces from removals
    text = re.sub(r'\s+', ' ', text).strip()
    # Add certainty insertion sometimes
    if random.random() < 0.3:
        ins = random.choice(OVERCONFIDENCE_INSERTIONS)
        text = ins + text[0].lower() + text[1:]
    return text


def apply_hedging(text: str) -> str:
    """Apply hedging to a sentence by prepending a hedging phrase."""
    prefix = random.choice(HEDGING_PREFIXES)
    # Avoid double-hedging
    if any(text.lower().startswith(p.lower().rstrip(" that ").rstrip()) for p in HEDGING_PREFIXES):
        return None
    return prefix + text[0].lower() + text[1:]


def apply_evidence_context(text: str) -> str:
    """Turn a bare claim into a CAL by prepending evidence context."""
    prefix = random.choice(EVIDENCE_PREFIXES)
    return prefix + text[0].lower() + text[1:]


def generate_synthetic_uc(record: dict) -> dict:
    """Generate a natural UNDERCLAIMING example by adding hedging."""
    text = record["text"]
    new_text = apply_hedging(text)
    if not new_text:
        return None
    # If original was already OC, tone it down to UC level
    certainty = "low"
    orig_label = record.get("label", -1)
    if orig_label == 0:
        certainty = "low"
    elif orig_label == 2:
        certainty = "low"
    else:
        certainty = "neutral"
    return {
        "text": new_text,
        "label": 1,
        "source": f"synthetic_uc_{record.get('source', 'unknown')}",
        "certainty": certainty,
        "original_label": "hedged_calibrated",
        "veracity_clarity": "unclear",
    }


def generate_synthetic_oc(record: dict) -> dict:
    """Generate a natural OVERCLAIMING example by amplifying certainty and removing hedges.

    Unlike the old approach (blatant trigger prefixes), this uses subtle transformations
    so the model learns to detect unwarranted certainty rather than surface trigger words.
    """
    text = record["text"]
    new_text = apply_overconfidence(text)
    if new_text == text:
        # Transformation had no effect — fall back to insertion
        ins = random.choice(OVERCONFIDENCE_INSERTIONS)
        new_text = ins + text[0].lower() + text[1:]
    return {
        "text": new_text,
        "label": 0,
        "source": f"synthetic_oc_{record.get('source', 'unknown')}",
        "certainty": "high",
        "original_label": "overconfident_calibrated",
        "veracity_clarity": "clear",
    }


def generate_hard_cal(record: dict) -> dict:
    """Generate a hard-negative CALIBRATED example by adding evidence context to an OC sentence.

    This teaches the model that scientific/technical language IS calibrated
    when accompanied by proper evidence qualifiers.
    """
    text = record["text"]
    new_text = apply_evidence_context(text)
    return {
        "text": new_text,
        "label": 2,
        "source": f"synthetic_hard_cal_{record.get('source', 'unknown')}",
        "certainty": "neutral",
        "original_label": "evidenced_overclaiming",
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
    # Always generate synthetic examples regardless of balance
    # The new subtle OC examples teach the model to detect unwarranted certainty
    # without relying on surface trigger words
    random.shuffle(cal_records)

    needed_uc = min(10000, len(cal_records) // 2)
    needed_oc_subtle = min(10000, len(cal_records) // 2)
    needed_hard_cal = min(10000, len(oc_records))

    uc_synthetic = []
    for r in cal_records:
        if len(uc_synthetic) >= needed_uc:
            break
        syn = generate_synthetic_uc(r)
        if syn:
            uc_synthetic.append(syn)
    all_records.extend(uc_synthetic)
    print(f"  Generated {len(uc_synthetic)} synthetic UNDERCLAIMING examples")

    oc_synthetic = []
    for r in cal_records[:needed_oc_subtle * 2]:
        if len(oc_synthetic) >= needed_oc_subtle:
            break
        syn = generate_synthetic_oc(r)
        if syn:
            oc_synthetic.append(syn)
    all_records.extend(oc_synthetic)
    print(f"  Generated {len(oc_synthetic)} subtle OVERCLAIMING examples")

    random.shuffle(oc_records)
    hard_cal = []
    for r in oc_records:
        if len(hard_cal) >= needed_hard_cal:
            break
        syn = generate_hard_cal(r)
        if syn:
            hard_cal.append(syn)
    all_records.extend(hard_cal)
    print(f"  Generated {len(hard_cal)} hard-negative CALIBRATED examples (OC + evidence context)")

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
