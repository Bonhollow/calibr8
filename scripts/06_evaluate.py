#!/usr/bin/env python3
"""
Step 6: Evaluate the trained model on test set using base + adapter (not fused).
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from mlx_lm import load, generate

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit-g32"
ADAPTER_PATH = Path(__file__).parent.parent / "adapters" / "calibr8"
HF_REPO = "Bonhollow/calibr8"
LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}
NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


def extract_label(text: str) -> str:
    text_upper = text.upper()
    for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
        if label in text_upper:
            return label
    return "UNKNOWN"


def load_test_data(path: Path, limit=None):
    inputs, labels = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            msgs = r.get("messages", [])
            if len(msgs) >= 2:
                user_text = msgs[0]["content"]
                assistant_text = msgs[1]["content"]
                label_str = extract_label(assistant_text)
                if label_str != "UNKNOWN":
                    inputs.append(user_text)
                    labels.append(NAME_TO_ID[label_str])
    return inputs, labels


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if np.any(in_bin):
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_prob[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    return ece


def main():
    print("=" * 60)
    print("STEP 6: Model evaluation (base + adapter, no fuse)")
    print("=" * 60)

    print("\nLoading base model + adapter...")
    t0 = time.time()
    if not ADAPTER_PATH.exists():
        print(f"Downloading Calibr8 adapter from {HF_REPO}...")
        import subprocess
        subprocess.run(["huggingface-cli", "download", HF_REPO, "--local-dir", str(ADAPTER_PATH)], check=True)
    model, tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    print(f"  Loaded in {time.time() - t0:.1f}s")

    limit = 423
    test_path = DATA_DIR / "test.jsonl"
    inputs, y_true = load_test_data(test_path, limit=limit)
    print(f"\nTest set: {len(inputs)} samples (limit={limit})")

    print("\nRunning inference...")
    y_pred = []
    y_probs = []
    start = time.time()

    for i, text in enumerate(inputs):
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=60,
        )

        label_str = extract_label(response)
        pred_id = NAME_TO_ID.get(label_str, -1)
        y_pred.append(pred_id)

        # Confidence proxy
        response_upper = response.upper()
        scores = {}
        for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
            scores[label] = 1.0 if label in response_upper else 0.0
        found = [l for l in scores if scores[l] > 0]
        if len(found) > 1:
            for l in found:
                scores[l] = 1.0 / len(found)
        probs = np.array([scores["OVERCLAIMING"], scores["UNDERCLAIMING"], scores["CALIBRATED"]])
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.array([1/3, 1/3, 1/3])
        y_probs.append(probs)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(inputs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(inputs)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Filter UNKNOWNs
    valid = [(t, p, prob) for t, p, prob in zip(y_true, y_pred, y_probs) if p != -1]
    if len(valid) < len(y_true):
        print(f"\n  Warning: {len(y_true) - len(valid)} UNKNOWN predictions (skipped)")

    y_true_val, y_pred_val, y_probs_val = zip(*valid) if valid else ([], [], [])
    y_true_arr = np.array(y_true_val)
    y_pred_arr = np.array(y_pred_val)
    y_probs_arr = np.array(y_probs_val)

    print(f"\n{'=' * 60}")
    print("CLASSIFICATION REPORT")
    print(f"{'=' * 60}")
    target_names = ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]
    print(classification_report(y_true_arr, y_pred_arr, target_names=target_names, digits=4))

    macro = f1_score(y_true_arr, y_pred_arr, average="macro")
    print(f"Macro F1: {macro:.4f} (target > 0.72)")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2])
    print(f"\nConfusion Matrix:")
    print(f"{'':>16} {'OC':>6} {'UC':>6} {'CAL':>6}")
    for i, name in enumerate(target_names):
        print(f"{name:>16}: {cm[i,0]:6d} {cm[i,1]:6d} {cm[i,2]:6d}")

    per_class = f1_score(y_true_arr, y_pred_arr, average=None)
    targets = {"OVERCLAIMING": 0.75, "UNDERCLAIMING": 0.60, "CALIBRATED": 0.75}
    print(f"\nPer-class F1:")
    for name, f1, in zip(target_names, per_class):
        arrow = "✓" if f1 >= targets[name] else ("△" if f1 >= 0.50 else "✗")
        print(f"  {name:15s}: {f1:.4f} (target {targets[name]}) {arrow}")

    y_true_bin = (y_true_arr == 2).astype(int)
    y_conf = y_probs_arr[:, 2]
    ece = compute_ece(y_true_bin, y_conf)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f} (target < 0.10)")

    acc = (y_true_arr == y_pred_arr).mean()
    print(f"Overall accuracy: {acc:.4f}")

    elapsed = time.time() - start
    print(f"\nTotal inference time: {elapsed:.0f}s ({elapsed/len(inputs):.1f}s/sample)")

    print(f"\n{'=' * 60}")
    print("Evaluation complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
