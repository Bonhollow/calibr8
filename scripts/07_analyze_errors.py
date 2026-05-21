#!/usr/bin/env python3
"""
Step 7: Error analysis — examine false predictions in detail.
Focus on: which texts trigger false OVERCLAIMING vs what gets missed.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from mlx_lm import load, generate

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_MODEL = "mlx-community/gemma-3-1b-it-qat-4bit"
ADAPTER_PATH = Path(__file__).parent.parent / "adapters" / "confidence_v2"
LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}
NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


def extract_label(text: str) -> str:
    text_upper = text.upper()
    for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
        if label in text_upper:
            return label
    return "UNKNOWN"


def main():
    print("=" * 60)
    print("STEP 7: Error analysis — confusion matrix breakdown")
    print("=" * 60)

    print("\nLoading base model + adapter...")
    t0 = time.time()
    model, tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    print(f"  Loaded in {time.time() - t0:.1f}s")

    limit = 500
    test_path = DATA_DIR / "test.jsonl"
    samples = []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            msgs = r.get("messages", [])
            if len(msgs) >= 2:
                text = msgs[1]["content"]
                label_str = extract_label(text)
                if label_str != "UNKNOWN":
                    samples.append((r, NAME_TO_ID[label_str]))

    print(f"\nRunning inference on {len(samples)} samples...")

    results = []  # (text, true_label, pred_label)
    start = time.time()
    for i, (r, true_label) in enumerate(samples):
        user_text = r["messages"][0]["content"]
        claim = user_text.split("\n\n", 1)[1]

        messages = [{"role": "user", "content": user_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        response = generate(model, tokenizer, prompt=formatted, max_tokens=60)
        pred_str = extract_label(response)
        pred_label = NAME_TO_ID.get(pred_str, -1)

        results.append((claim, true_label, pred_label, response[:200]))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    print(f"\nInference done: {elapsed:.0f}s total ({elapsed/len(samples):.1f}s/sample)")

    # Group by confusion category
    categories = defaultdict(list)
    for claim, true_l, pred_l, raw in results:
        categories[(true_l, pred_l)].append((claim, raw))

    print(f"\n{'=' * 60}")
    print("CONFUSION MATRIX BREAKDOWN")
    print(f"{'=' * 60}")

    true_names = ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]
    print(f"{'':>16}", end="")
    for p in ["OC", "UC", "CAL"]:
        print(f"{p:>8}", end="")
    print()

    counts = defaultdict(int)
    for claim, true_l, pred_l, _ in results:
        counts[(true_l, pred_l)] += 1

    for t_name in true_names:
        print(f"{t_name:>16}", end="")
        for p_name in true_names:
            k = (NAME_TO_ID[t_name], NAME_TO_ID[p_name])
            print(f"{counts[k]:>8d}", end="")
        print()

    # Total
    print(f"{'TOTAL':>16}", end="")
    for p_name in true_names:
        total_p = sum(counts[(t, NAME_TO_ID[p_name])] for t in [0, 1, 2])
        print(f"{total_p:>8d}", end="")
    print()

    # Accuracy
    correct = sum(1 for _, t, p, _ in results if t == p)
    unknown = sum(1 for _, t, p, _ in results if p == -1)
    print(f"\nAccuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"UNKNOWN: {unknown}/{len(results)} ({100*unknown/len(results):.1f}%)")

    # --- False positives: CALIBRATED predicted as OVERCLAIMING ---
    print(f"\n{'=' * 60}")
    print("FALSE POSITIVES: CALIBRATED → OVERCLAIMING")
    print(f"{'=' * 60}")
    fp_samples = categories[(2, 0)]
    print(f"Count: {len(fp_samples)}")
    for i, (claim, raw) in enumerate(fp_samples[:20]):
        print(f"\n  [{i+1}] {claim[:120]}")
        print(f"      Model: {raw[:100]}")

    # --- False negatives: OVERCLAIMING predicted as CALIBRATED ---
    print(f"\n{'=' * 60}")
    print("FALSE NEGATIVES: OVERCLAIMING → CALIBRATED")
    print(f"{'=' * 60}")
    fn_samples = categories[(0, 2)]
    print(f"Count: {len(fn_samples)}")
    for i, (claim, raw) in enumerate(fn_samples[:20]):
        print(f"\n  [{i+1}] {claim[:120]}")
        print(f"      Model: {raw[:100]}")

    # --- CORRECT OVERCLAIMING ---
    print(f"\n{'=' * 60}")
    print("CORRECT: OVERCLAIMING → OVERCLAIMING")
    print(f"{'=' * 60}")
    oc_correct = categories[(0, 0)]
    print(f"Count: {len(oc_correct)}")
    for i, (claim, raw) in enumerate(oc_correct[:10]):
        print(f"\n  [{i+1}] {claim[:120]}")
        print(f"      Model: {raw[:80]}")

    # --- UNDERCLAIMING cases ---
    print(f"\n{'=' * 60}")
    print("ALL UNDERCLAIMING CASES (true or predicted)")
    print(f"{'=' * 60}")
    uc_cases = [(c, t, p, r) for c, t, p, r in results if t == 1 or p == 1]
    for i, (claim, true_l, pred_l, raw) in enumerate(uc_cases[:20]):
        true_name = LABEL_NAMES[true_l]
        pred_name = LABEL_NAMES.get(pred_l, "UNKNOWN")
        mark = "✓" if true_l == pred_l else "✗"
        print(f"\n  [{i+1}] True={true_name} Pred={pred_name} {mark}")
        print(f"      {claim[:150]}")
        print(f"      Raw: {raw[:120]}")

    # --- Over-prediction analysis ---
    print(f"\n{'=' * 60}")
    print("BIAS ANALYSIS")
    print(f"{'=' * 60}")

    true_oc = sum(1 for _, t, _, _ in results if t == 0)
    pred_oc = sum(1 for _, _, p, _ in results if p == 0)
    true_cal = sum(1 for _, t, _, _ in results if t == 2)
    pred_cal = sum(1 for _, _, p, _ in results if p == 2)
    true_uc = sum(1 for _, t, _, _ in results if t == 1)
    pred_uc = sum(1 for _, _, p, _ in results if p == 1)

    print(f"\n  OVERCLAIMING: true={true_oc}, predicted={pred_oc} ({'over' if pred_oc > true_oc else 'under'} by {abs(pred_oc-true_oc)})")
    print(f"  UNDERCLAIMING: true={true_uc}, predicted={pred_uc} ({'over' if pred_uc > true_uc else 'under'} by {abs(pred_uc-true_uc)})")
    print(f"  CALIBRATED:    true={true_cal}, predicted={pred_cal} ({'over' if pred_cal > true_cal else 'under'} by {abs(pred_cal-true_cal)})")

    # Check for common trigger words in FPs
    print(f"\n  Common words/phrases in FALSE POSITIVES (CAL → OC):")
    from collections import Counter
    word_counts = Counter()
    for claim, _ in fp_samples:
        words = claim.lower().split()
        # Skip very common words
        skip = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to",
                "and", "for", "with", "on", "by", "that", "this", "at", "from",
                "or", "as", "be", "we", "it", "not", "but", "all", "no", "had",
                "has", "have", "been", "its", "their", "our", "they", "these",
                "those", "can", "may", "will", "per", "than", "who", "which",
                "what", "when", "where", "how", "more", "most", "some", "any",
                "such", "each", "both", "also", "only", "very", "just", "about",
                "up", "out", "if", "so", "do", "does", "did", "shown", "using",
                "used", "due", "after", "before", "between", "among", "over",
                "under", "while", "during", "through", "other", "many", "much",
                "one", "two", "new", "first", "last", "high", "low", "long",
                "large", "small", "than", "into", "within", "without"}
        for w in words:
            if w not in skip and len(w) > 3:
                word_counts[w] += 1

    for word, count in word_counts.most_common(30):
        print(f"    {word:20s}: {count}")


if __name__ == "__main__":
    main()
