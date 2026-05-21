#!/usr/bin/env python3
"""
Step 1: Download raw datasets to data/raw/ as JSONL with consistent fields.

Downloads:
  - LIAR-PLUS:    Political statements with 6-class veracity labels (~10K records)
  - AVeriTeC:     Claims with veracity + evidence (~3K train + dev)
  - SciFact:      Scientific claims vs abstracts (~1.4K records)
  - HealthVer:    Health claims with evidence labels (~14K records)
  - ClaimBuster:  News sentences rated by claim-worthiness (~23K records)
  - FEVER:        Large-scale claim verification dataset (~185K records)
  - YMETHO:       Central bank minutes with certainty labels (~25K records)
"""

import json
import os
import csv
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_bytes(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return response.read()


def download_url_to_string(url):
    return download_bytes(url).decode('utf-8')


def download_liar():
    """LIAR-PLUS — JSONL format from GitHub (TSV moved to subdir, JSONL is cleaner)."""
    out_path = RAW_DIR / "liar.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading LIAR-PLUS from GitHub (JSONL)...")
    splits = {
        "train": "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/jsonl/train2.jsonl",
        "validation": "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/jsonl/val2.jsonl",
        "test": "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/jsonl/test2.jsonl",
    }

    records = []
    for split_name, url in splits.items():
        try:
            content = download_url_to_string(url)
            for line in content.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("claim", "").strip()
                label = row.get("label", "").lower()
                if text and label:
                    records.append({
                        "text": text,
                        "original_label": label,
                        "source": "liar",
                        "split": split_name,
                    })
        except Exception as e:
            print(f"    [WARN] LIAR {split_name} split: {e}")

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  [DONE] LIAR: {len(records)} records → {out_path.name}")
    else:
        print("  [ERROR] No LIAR records downloaded")


def download_averitec():
    """AVeriTeC — train + dev from GitHub (test.json is held out, returns 404)."""
    out_path = RAW_DIR / "averitec.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading AVeriTeC from GitHub...")
    splits = {
        "train": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json",
        "dev": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json",
    }

    records = []
    for split_name, url in splits.items():
        try:
            data = json.loads(download_url_to_string(url))
            for row in data:
                text = row.get("claim", "").strip()
                label = str(row.get("label", row.get("verdict", "unknown"))).lower()
                if text:
                    records.append({
                        "text": text,
                        "original_label": label,
                        "source": "averitec",
                        "split": "validation" if split_name == "dev" else split_name,
                    })
        except Exception as e:
            print(f"    [WARN] AVeriTeC {split_name} split: {e}")

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        counts = Counter(r["split"] for r in records)
        print(f"  [DONE] AVeriTeC: {len(records)} records → {out_path.name} ({dict(counts)})")
    else:
        print("  [ERROR] No AVeriTeC records downloaded")


def download_scifact():
    """SciFact — S3 tarball (GitHub raw URLs restructured, S3 still works)."""
    out_path = RAW_DIR / "scifact.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading SciFact from S3 tarball...")
    url = "https://scifact.s3.us-west-2.amazonaws.com/release/latest/data.tar.gz"

    try:
        data = download_bytes(url)
    except Exception as e:
        print(f"    [ERROR] Failed to download SciFact tarball: {e}")
        return

    records = []
    split_map = {"claims_train.jsonl": "train", "claims_dev.jsonl": "validation", "claims_test.jsonl": "test"}

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        tmp.write(data)
        tmp.flush()
        with tarfile.open(tmp.name, "r:gz") as tar:
            for member_path, split_name in split_map.items():
                try:
                    f = tar.extractfile(f"data/{member_path}")
                    if not f:
                        continue
                    for line in f:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        claim = row.get("claim", "").strip()
                        evidence_label = "unknown"
                        evidence_dict = row.get("evidence", {})
                        if evidence_dict:
                            labels = []
                            for doc_id, ev_list in evidence_dict.items():
                                for ev in ev_list:
                                    if "label" in ev:
                                        labels.append(ev["label"].lower())
                            if "support" in labels:
                                evidence_label = "support"
                            elif "contradict" in labels:
                                evidence_label = "contradict"
                        if claim:
                            records.append({
                                "text": claim,
                                "original_label": evidence_label,
                                "source": "scifact",
                                "split": split_name,
                            })
                except Exception as e:
                    print(f"    [WARN] SciFact {member_path}: {e}")

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        counts = Counter(r["split"] for r in records)
        print(f"  [DONE] SciFact: {len(records)} records → {out_path.name} ({dict(counts)})")
    else:
        print("  [ERROR] No SciFact records downloaded")


def download_healthver():
    """HealthVer — CSV files from GitHub (SUPPORT/REFUTE/NEUTRAL health claims)."""
    out_path = RAW_DIR / "healthver.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading HealthVer from GitHub...")
    splits = {
        "train": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_train.csv",
        "validation": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_dev.csv",
        "test": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_test.csv",
    }

    records = []
    for split_name, url in splits.items():
        try:
            content = download_url_to_string(url)
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                text = row.get("claim", "").strip()
                label = row.get("label", "").lower().strip()
                if text and label:
                    records.append({
                        "text": text,
                        "original_label": label,
                        "source": "healthver",
                        "split": split_name,
                    })
        except Exception as e:
            print(f"    [WARN] HealthVer {split_name} split: {e}")

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        counts = Counter(r["split"] for r in records)
        print(f"  [DONE] HealthVer: {len(records)} records → {out_path.name} ({dict(counts)})")
    else:
        print("  [ERROR] No HealthVer records downloaded")


def download_claimbuster():
    """ClaimBuster — parquet from HuggingFace (Zenodo 403, HF parquet accessible)."""
    out_path = RAW_DIR / "claimbuster.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading ClaimBuster from HuggingFace (parquet)...")
    url = ("https://huggingface.co/datasets/Nithiwat/claimbuster/resolve/main/"
           "data/train-00000-of-00001-5f01fa457a7cb014.parquet")

    try:
        import pandas as pd
    except ImportError:
        print("    [ERROR] pandas required for ClaimBuster. Install with: pip install pandas")
        return

    try:
        data = download_bytes(url)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(data)
            tmp.flush()
            df = pd.read_parquet(tmp.name)
    except Exception as e:
        print(f"    [ERROR] Failed to download/parse ClaimBuster: {e}")
        return

    records = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        label = str(row.get("checkworthiness", "0"))
        if text:
            records.append({
                "text": text,
                "original_label": label,
                "source": "claimbuster",
                "split": "train",
            })

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  [DONE] ClaimBuster: {len(records)} records → {out_path.name}")
    else:
        print("  [ERROR] No ClaimBuster records downloaded")


def download_fever():
    """FEVER — large-scale claim verification from fever.ai (SUPPORTS/REFUTES/NOTENOUGHINFO)."""
    out_path = RAW_DIR / "fever.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading FEVER (185K claims — this may take a moment)...")
    url = "https://fever.ai/download/fever/train.jsonl"

    try:
        content = download_url_to_string(url)
    except Exception as e:
        print(f"    [ERROR] Failed to download FEVER: {e}")
        return

    records = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            text = row.get("claim", "").strip()
            label = row.get("label", "").lower().replace(" ", "")
            if text and label in ("supports", "refutes", "notenoughinfo"):
                records.append({
                    "text": text,
                    "original_label": label,
                    "source": "fever",
                    "split": "train",
                })
        except json.JSONDecodeError:
            continue

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  [DONE] FEVER: {len(records)} records → {out_path.name}")
    else:
        print("  [ERROR] No FEVER records downloaded")


def download_ymetho():
    """YMETHO — central bank minutes with certainty labels from HuggingFace parquet."""
    out_path = RAW_DIR / "ymetho.jsonl"
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    print("  Downloading YMETHO from HuggingFace (parquet)...")
    url = ("https://huggingface.co/datasets/YMETHO/all_annotated_sentences_25000/"
           "resolve/main/5768/train-00000-of-00001.parquet")

    try:
        import pandas as pd
    except ImportError:
        print("    [ERROR] pandas required for YMETHO. Install with: pip install pandas")
        return

    try:
        data = download_bytes(url)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(data)
            tmp.flush()
            df = pd.read_parquet(tmp.name)
    except Exception as e:
        print(f"    [ERROR] Failed to download/parse YMETHO: {e}")
        return

    records = []
    for _, row in df.iterrows():
        text = str(row.get("sentences", "")).strip()
        label = str(row.get("certain_label", "")).lower().strip()
        if text and len(text) > 20 and label in ("certain", "uncertain"):
            records.append({
                "text": text,
                "original_label": label,
                "source": "ymetho",
                "split": "train",
            })

    if records:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  [DONE] YMETHO: {len(records)} records → {out_path.name}")
    else:
        print("  [ERROR] No YMETHO records downloaded")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STEP 1: Downloading datasets to data/raw/")
    print("=" * 60)

    download_liar()
    download_averitec()
    download_scifact()
    download_healthver()
    download_claimbuster()
    download_fever()
    download_ymetho()

    print("\n" + "=" * 60)
    print("Download summary:")
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(f))
        print(f"  {f.name}: {count} records")
    print("=" * 60)


if __name__ == "__main__":
    main()
