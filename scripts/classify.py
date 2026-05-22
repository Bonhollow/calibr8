#!/usr/bin/env python3
"""
Confidence Calibration Detector — Inference Pipeline.

Usage:
  python3 scripts/classify.py "Studies prove this supplement cures inflammation."
  python3 scripts/classify.py --batch batch.txt
  python3 scripts/classify.py --json '{"text": "..."}'

Output: structured JSON with label, confidence, explanation, and span.
"""

import json
import re
import sys
import argparse
from pathlib import Path

from mlx_lm import load, generate

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit-g32"
HF_REPO = "Bonhollow/calibr8"
ADAPTER_PATH = Path(__file__).parent.parent / "adapters" / "calibr8"

LABEL_NAMES = {0: "OVERCLAIMING", 1: "UNDERCLAIMING", 2: "CALIBRATED"}

OVERCONFIDENCE_SIGNALS = [
    r"\bdefinitively\b", r"\bproven\b", r"\bproves\b", r"\bprove\b",
    r"\balways\b", r"\bnever\b", r"\bconclusively\b", r"\bundeniably\b",
    r"\bwithout a doubt\b", r"\birrefutable\b", r"\bobviously\b",
    r"\bdefinitive\b", r"\bno doubt\b", r"\bbeyond question\b",
]

UNDERCONFIDENCE_SIGNALS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\blikely\b", r"\bunlikely\b", r"\bpotentially\b", r"\bsuggests?\b",
    r"\bappears?\b", r"\bseems?\b", r"\bindicates?\b", r"\btentative\b",
    r"\bpreliminary\b", r"\bunclear\b", r"\buncertain\b",
]

_model = None
_tokenizer = None


def _ensure_adapter():
    if not ADAPTER_PATH.exists():
        print(f"Downloading Calibr8 adapter from {HF_REPO}...")
        import subprocess
        subprocess.run(
            ["huggingface-cli", "download", HF_REPO, "--local-dir", str(ADAPTER_PATH)],
            check=True
        )

def _load():
    global _model, _tokenizer
    if _model is None:
        _ensure_adapter()
        _model, _tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    return _model, _tokenizer


def extract_label(text: str) -> str:
    text_upper = text.upper()
    for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
        if label in text_upper:
            return label
    return "UNKNOWN"


def extract_confidence(text: str) -> float:
    m = re.search(r"\((\d+\.\d+)\)", text)
    if m:
        return float(m.group(1))
    return None


def extract_span(text: str) -> str:
    m = re.search(r'Span:\s*"([^"]*)"', text)
    if m:
        return m.group(1)
    return None


def extract_explanation(text: str) -> str:
    parts = text.split(". ", 1)
    if len(parts) > 1:
        explanation = parts[1]
        span_m = re.search(r'Span:\s*"[^"]*"', explanation)
        if span_m:
            explanation = explanation[:span_m.start()].strip()
        return explanation if len(explanation) > 5 else None
    return None


def fallback_span(text: str, label: str) -> str:
    if label == "OVERCLAIMING":
        for p in OVERCONFIDENCE_SIGNALS:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return text[m.start():m.end()]
        return "the claim"
    if label == "UNDERCLAIMING":
        for p in UNDERCONFIDENCE_SIGNALS:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return text[m.start():m.end()]
        return "hedging language"
    return None


def classify(text: str) -> dict:
    model, tokenizer = _load()

    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    response = generate(model, tokenizer, prompt=formatted, max_tokens=60)

    label = extract_label(response)
    confidence = extract_confidence(response)
    span = extract_span(response) or fallback_span(text, label)
    explanation = extract_explanation(response)

    return {
        "label": label,
        "confidence": confidence if confidence is not None else 0.5,
        "span": span if span != "the claim" and span != "hedging language" else None,
        "explanation": explanation or "",
        "raw_response": response,
    }


def classify_batch(texts: list[str]) -> list[dict]:
    return [classify(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="Confidence Calibration Detector")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--json", help="JSON string with {text: ...}")
    parser.add_argument("--batch", help="File with one text per line")
    parser.add_argument("--pretty", action="store_true", default=True, help="Pretty print JSON output")
    args = parser.parse_args()

    if args.batch:
        with open(args.batch) as f:
            texts = [line.strip() for line in f if line.strip()]
        results = classify_batch(texts)
    elif args.json:
        data = json.loads(args.json)
        texts = data if isinstance(data, list) else [data["text"]]
        results = classify_batch(texts)
    elif args.text:
        results = [classify(args.text)]
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(results, indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    main()
