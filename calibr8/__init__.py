"""
calibr8 — Confidence Calibration Detector.

Usage:
    from calibr8 import classify, load

    result = classify("Studies prove this cures inflammation.")
    # => {"label": "OVERCLAIMING", "confidence": 0.78, ...}

    model, tokenizer = load()
    # => use directly with mlx_lm.generate()
"""

import json
import re
import subprocess
from pathlib import Path

from mlx_lm import load as _mlx_load, generate

__version__ = "1.0.0"

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit-g32"
HF_REPO = "Bonhollow/calibr8"
ADAPTER_DIR = Path.home() / ".cache" / "calibr8"

_model = None
_tokenizer = None

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


def _ensure_adapter() -> Path:
    adapter_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_path.exists():
        print(f"Downloading calibr8 adapter from {HF_REPO}...")
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["hf", "download", HF_REPO, "--local-dir", str(ADAPTER_DIR)],
                check=True,
            )
        except FileNotFoundError:
            subprocess.run(
                ["huggingface-cli", "download", HF_REPO, "--local-dir", str(ADAPTER_DIR)],
                check=True,
            )
        if not adapter_path.exists():
            raise RuntimeError(
                f"Failed to download adapter from {HF_REPO}. "
                f"Try: hf download {HF_REPO} --local-dir {ADAPTER_DIR}"
            )
    return ADAPTER_DIR


def load():
    """Load the calibr8 model and tokenizer. Returns (model, tokenizer)."""
    global _model, _tokenizer
    if _model is None:
        adapter_path = _ensure_adapter()
        _model, _tokenizer = _mlx_load(BASE_MODEL, adapter_path=str(adapter_path))
    return _model, _tokenizer


def unload():
    """Free model memory. Next call to load() will re-download."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None


def extract_label(text: str) -> str:
    text_upper = text.upper()
    for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
        if label in text_upper:
            return label
    return "UNKNOWN"


def extract_confidence(text: str) -> float | None:
    m = re.search(r"\((\d+\.\d+)\)", text)
    return float(m.group(1)) if m else None


def extract_span(text: str) -> str | None:
    m = re.search(r'Span:\s*"([^"]*)"', text)
    return m.group(1) if m else None


def extract_explanation(text: str) -> str | None:
    parts = text.split(". ", 1)
    if len(parts) > 1:
        explanation = parts[1]
        span_m = re.search(r'Span:\s*"[^"]*"', explanation)
        if span_m:
            explanation = explanation[:span_m.start()].strip()
        return explanation if len(explanation) > 5 else None
    return None


def fallback_span(text: str, label: str) -> str | None:
    signals = OVERCONFIDENCE_SIGNALS if label == "OVERCLAIMING" else (
        UNDERCONFIDENCE_SIGNALS if label == "UNDERCLAIMING" else []
    )
    for p in signals:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return text[m.start():m.end()]
    return "the claim" if label == "OVERCLAIMING" else "hedging language" if label == "UNDERCLAIMING" else None


def classify(text: str) -> dict:
    """Classify a single text. Returns {label, confidence, span, explanation, raw_response}."""
    model, tokenizer = load()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=60)

    label = extract_label(response)
    confidence = extract_confidence(response) or 0.5
    span = extract_span(response) or fallback_span(text, label)
    explanation = extract_explanation(response) or ""
    span = span if span not in ("the claim", "hedging language") else None

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "span": span,
        "explanation": explanation,
        "raw_response": response,
    }


def classify_batch(texts: list[str]) -> list[dict]:
    """Classify multiple texts sequentially."""
    return [classify(t) for t in texts]
