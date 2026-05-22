"""
CLI entry point for calibr8.

Usage:
    python3 -m calibr8 "Studies prove this cures inflammation."
    python3 -m calibr8 --batch texts.txt
    echo '{"text": "..."}' | python3 -m calibr8 --json
"""

import json
import sys
import argparse

from calibr8 import classify, classify_batch


def main():
    parser = argparse.ArgumentParser(
        description="calibr8 — Confidence Calibration Detector"
    )
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--batch", help="File with one text per line")
    parser.add_argument("--json", help="JSON string or pipe with {text: ...}")
    parser.add_argument("--pretty", action="store_true", default=True)
    parser.add_argument("--minimal", action="store_true",
                        help="Output just the label (no JSON)")
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

    if args.minimal:
        for r in results:
            print(r["label"])
    else:
        print(json.dumps(results, indent=2 if args.pretty else None,
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
