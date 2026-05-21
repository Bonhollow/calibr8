#!/usr/bin/env python3
"""
Step 2: Remap original dataset labels to the 3-class confidence calibration scheme.

Uses regex-based certainty/hedging detection + original veracity labels.
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
REMAPPED_DIR = Path(__file__).parent.parent / "data" / "remapped"

OVERCONFIDENCE_SIGNALS = [
    r"\bdefinitively\b", r"\bproven\b", r"\bproves\b", r"\bprove\b",
    r"\balways\b", r"\bnever\b", r"\b100%\b", r"\bconclusively\b",
    r"\bundeniably\b", r"\bfact\b", r"\bscience has shown\b",
    r"\bstudies prove\b", r"\bdefinitely\b", r"\bcertainly\b",
    r"\babsolutely\b", r"\bundoubtedly\b", r"\bwithout a doubt\b",
    r"\bguaranteed?\b", r"\birrefutable\b", r"\bclearly\b", r"\bobviously\b",
]

UNDERCONFIDENCE_SIGNALS = [
    # Multi-word hedging phrases
    r"\bsome suggest\b", r"\bsome researchers think\b",
    r"\bmight possibly\b", r"\bcould perhaps\b",
    r"\bit is conceivable\b", r"\bit is possible\b",
    r"\bsome evidence\b", r"\bnot necessarily\b",
    r"\bsubject to\b", r"\bdownside risk\b",
    r"\bon balance\b", r"\bin general\b",
    r"\bto some extent\b", r"\btends? to\b",
    # Modal verbs of uncertainty
    r"\bmay\b", r"\bmight\b", r"\bcould\b",
    # Epistemic adverbs
    r"\bpossibly\b", r"\bperhaps\b", r"\bmaybe\b",
    r"\blikely\b", r"\bunlikely\b",
    r"\bpotentially\b", r"\bpotential\b",
    r"\blargely\b", r"\bbroadly\b", r"\bgenerally\b",
    r"\brelatively\b", r"\bsomewhat\b",
    # Hedging verbs
    r"\bsuggest(?:s|ed|ing)?\b",
    r"\bexpect(?:ed|s|ing)?\b",
    r"\bappears?\b", r"\bseems?\b",
    r"\bindicates?\b", r"\bindicated\b",
    r"\btentative\b", r"\bpreliminary\b",
    r"\bunclear\b", r"\buncertain\b",
]

VERACITY_MAP = {
    "true": "supported", "mostly-true": "supported",
    "half-true": "disputed", "barely-true": "refuted",
    "false": "refuted", "pants-fire": "refuted",
    "supported": "supported", "refuted": "refuted",
    "not enough evidence": "disputed",
    "conflicting evidence/cherrypicking": "disputed",
    "support": "supported", "contradict": "refuted",
    "noinfo": "disputed", "unknown": "disputed",
    "mixture": "disputed", "mostly true": "supported",
    "mostly false": "refuted", "unproven": "disputed",
    "1": "supported", "0": "refuted", "2": "disputed",
    "supports": "supported", "refutes": "refuted", "neutral": "disputed",
    "notenoughinfo": "disputed",
    "certain": "supported", "uncertain": "supported",
}

LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}


def detect_certainty_level(text: str) -> str:
    text_lower = text.lower()
    over = sum(1 for p in OVERCONFIDENCE_SIGNALS if re.search(p, text_lower))
    under = sum(1 for p in UNDERCONFIDENCE_SIGNALS if re.search(p, text_lower))
    if over > under and over >= 1:
        return "high"
    elif under > over and under >= 1:
        return "low"
    return "neutral"


def remap_label(original_label: str, text: str) -> int:
    veracity = VERACITY_MAP.get(original_label.lower().strip(), "disputed")
    certainty = detect_certainty_level(text)

    if veracity == "supported":
        return 1 if certainty == "low" else 2
    elif veracity == "refuted":
        return 2 if certainty == "low" else 0
    else:  # disputed
        return 0 if certainty == "high" else 2


def process_file(input_path: Path, output_path: Path) -> dict:
    stats = {0: 0, 1: 0, 2: 0, "total": 0, "skipped": 0}
    records = []
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text or len(text) < 10:
                stats["skipped"] += 1
                continue
            label = remap_label(row.get("original_label", "unknown"), text)
            records.append({
                "text": text, "label": label,
                "label_name": LABEL_NAMES[label],
                "original_label": row.get("original_label", "unknown"),
                "source": row.get("source", input_path.stem),
                "certainty": detect_certainty_level(text),
            })
            stats[label] += 1
            stats["total"] += 1

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return stats


def main():
    REMAPPED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STEP 2: Remapping labels to calibration classes")
    print("=" * 60)
    total = {0: 0, 1: 0, 2: 0, "total": 0, "skipped": 0}
    for raw_file in sorted(RAW_DIR.glob("*.jsonl")):
        out_file = REMAPPED_DIR / raw_file.name
        print(f"\n  Processing {raw_file.name}...")
        stats = process_file(raw_file, out_file)
        print(f"    Total: {stats['total']}  |  OC: {stats[0]}  |  UC: {stats[1]}  |  CAL: {stats[2]}  |  Skip: {stats['skipped']}")
        for k in total:
            total[k] += stats[k]
    print(f"\n{'=' * 60}")
    print(f"GRAND TOTAL: {total['total']}  |  OC: {total[0]}  |  UC: {total[1]}  |  CAL: {total[2]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
