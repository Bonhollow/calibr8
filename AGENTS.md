# Confidence Calibration Detector — MLX Build Plan

## What we are building

A ~1B parameter model that reads a sentence or paragraph and classifies it as:
- **OVERCLAIMING** — certainty exceeds the evidence (e.g. "studies definitively prove...")
- **UNDERCLAIMING** — excessive hedging where evidence is actually strong
- **CALIBRATED** — expressed certainty matches the evidence

Secondary output: a confidence score (0–1) and a span highlight of the miscalibrated phrase.

---

## Hardware assumptions

- Apple Silicon Mac (M1/M2/M3/M4) with at least 16 GB unified memory
- MLX + mlx-lm as the training and inference framework
- No CUDA, no cloud required

---

## Phase 1 — Dataset

### 1.1 Primary sources (all free)

| Source | What it provides | Size | Label availability |
|---|---|---|---|---|
| [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) | Political statements with 6-class veracity labels | ~10K | Partially usable |
| [AVeriTeC](https://github.com/MichSchli/AVeriTeC) | Claims with veracity + evidence | ~3.5K train+dev (test held out) | Strong signal |
| [SciFact](https://scifact.s3.us-west-2.amazonaws.com/release/latest/data.tar.gz) | Scientific claims vs abstracts | 1.4K | Gold standard for science |
| [HealthVer](https://github.com/sarrouti/HealthVer) | Health claims with evidence (SUPPORT/REFUTE/NEUTRAL) | 14K | High value |
| [ClaimBuster](https://huggingface.co/datasets/Nithiwat/claimbuster) | News sentences rated by claim-worthiness | 23K | Noisy but large |
| [FEVER](https://fever.ai/dataset/fever.html) | Large-scale claim verification (SUPPORTS/REFUTES/NEI) | 185K | Large, great for diversification |
| [YMETHO](https://huggingface.co/datasets/YMETHO/all_annotated_sentences_25000) | Central bank minutes with certainty labels (uncertain/certain) | 25K | High value — rich hedging language |
| [WiCE](https://huggingface.co/datasets/Harvard-SEAS-NLP/wice) | Textual entailment fine-grained | 8K | Useful for span detection |
| SEC earnings calls | Corporate hedging language | Unlimited | Must label yourself |

> **Note:** The `01_download_datasets.py` script now uses direct raw URLs (not the HuggingFace `datasets` library, which has issues with script-based datasets). LIAR-PLUS uses GitHub JSONL, SciFact uses an S3 tarball, HealthVer uses GitHub CSV, ClaimBuster uses HuggingFace parquet, FEVER uses a direct download, and YMETHO uses HuggingFace parquet. AVeriTeC test split is held out and skipped.

### 1.2 Label mapping strategy

Most datasets use fact-check style labels (true/false/misleading). Remap to your 3 classes:

```
SUPPORTED + high certainty language  → CALIBRATED
SUPPORTED + hedged language          → UNDERCLAIMING
REFUTED + high certainty language    → OVERCLAIMING
REFUTED + hedged language            → CALIBRATED (wrong but at least honest)
DISPUTED or MIXTURE                  → OVERCLAIMING if phrased with certainty
```

Use a simple regex pass first to detect certainty language before remapping:

```python
OVERCONFIDENCE_SIGNALS = [
    r"\bdefinitively\b", r"\bproven\b", r"\balways\b", r"\bnever\b",
    r"\b100%\b", r"\bconclusively\b", r"\bundeniably\b", r"\bfact\b",
    r"\bscience has shown\b", r"\bstudies prove\b"
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
```

### 1.3 Synthetic augmentation

Generate hard negatives using Claude via API — this is legitimate for training data:

```python
AUGMENTATION_PROMPT = """
Given this calibrated sentence, rewrite it in two ways:
1. Overclaiming version — add unwarranted certainty
2. Underclaiming version — add excessive hedging

Sentence: {sentence}

Return JSON: {{"overclaiming": "...", "underclaiming": "...", "original": "..."}}
"""
```

Target: 3,000–5,000 synthetic triples across diverse domains (science, health, finance, politics).

### 1.4 Final dataset composition

| Split | Size | Notes |
|---|---|---|
| Train | ~18,000 | Balanced across 3 classes |
| Validation | ~2,000 | Held out by source, not random |
| Test | ~1,000 | Never touched until final eval |

Hold out by source (e.g. all SciFact in test) to measure real generalization, not memorization.

### 1.5 Dataset format

Save as JSONL for mlx-lm compatibility:

```jsonl
{"text": "Studies have definitively proven this supplement cures inflammation.", "label": 0}
{"text": "Some early research suggests there may be a possible link.", "label": 1}
{"text": "A 2023 RCT found significant improvement in 68% of participants.", "label": 2}
```

Label map: `0 = OVERCLAIMING`, `1 = UNDERCLAIMING`, `2 = CALIBRATED`

For instruction-tuned fine-tuning (preferred), use chat format:

```jsonl
{"messages": [
  {"role": "user", "content": "Classify the confidence calibration of this text:\n\nStudies have definitively proven this supplement cures inflammation."},
  {"role": "assistant", "content": "OVERCLAIMING. The word 'definitively proven' signals absolute certainty. No single study can definitively prove a health claim; the appropriate phrasing would be 'evidence suggests' or 'a study found'."}
]}
```

---

## Phase 2 — Model selection

### Recommended base models (all MLX-compatible)

| Model | Params | MLX repo | Why |
|---|---|---|---|
| **Gemma 3 1B IT (QAT)** | 1B | `mlx-community/gemma-3-1b-it-qat-4bit` | Google's latest, QAT retains quality at 4-bit, excellent instruction following |
| Gemma 3 1B IT | 1B | `mlx-community/gemma-3-1b-it-4bit` | Same architecture, standard post-hoc quantization |
| Qwen3-0.6B | 0.6B | `mlx-community/Qwen3-0.6B-4bit` | Ultra-lightweight, newest Qwen generation, fastest iteration |
| Qwen2.5-1.5B-Instruct | 1.5B | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Strong reasoning, good fallback |

**Start with Gemma 3 1B IT QAT-4bit.** The QAT (quantization-aware training) means the 4-bit weights are trained to handle quantization, retaining more quality than post-hoc quantized models. Excellent instruction following for classification tasks.

---

## Phase 3 — Fine-tuning with MLX-LM

### 3.1 Install

```bash
pip install mlx-lm
```

### 3.2 Prepare data

mlx-lm expects data in `data/train.jsonl`, `data/valid.jsonl`:

```bash
mkdir data
# copy your prepared JSONL files here
```

### 3.3 LoRA fine-tuning config

Create `lora_config.yaml`:

```yaml
model: mlx-community/gemma-3-1b-it-qat-4bit

# LoRA settings
lora_layers: 16          # number of transformer layers to adapt
lora_rank: 8             # rank of LoRA matrices — start low, increase if underfitting
lora_alpha: 16           # scaling factor, usually 2x rank
lora_dropout: 0.05

# Training
batch_size: 4
iters: 2000              # adjust based on dataset size
val_batches: 25
learning_rate: 1e-5      # conservative for instruction-tuned base
lr_schedule: cosine_decay
warmup_steps: 100
grad_checkpoint: true    # saves memory on 16 GB

# Data
train: data/train.jsonl
valid: data/valid.jsonl
max_seq_length: 512      # confidence claims are short, 512 is enough

# Output
adapter_path: adapters/confidence_v1
save_every: 200
```

### 3.4 Run training

```bash
mlx_lm.lora \
  --config lora_config.yaml \
  --train
```

Training 2000 iters on ~18K samples takes roughly 45–90 minutes on M2/M3, depending on batch size and sequence length.

### 3.5 Monitor training

Watch for:
- Validation loss decreasing and stabilizing (not diverging)
- Train loss should be lower than val loss but not by a huge margin (overfitting signal)
- If val loss stops improving after ~800 iters, stop early

```bash
# Loss curve is printed to stdout — pipe to a file and plot
mlx_lm.lora --config lora_config.yaml --train 2>&1 | tee training_log.txt
```

### 3.6 Fuse the adapter

After training, merge LoRA weights back into the base model for clean inference:

```bash
mlx_lm.fuse \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --adapter-path adapters/confidence_v1 \
  --save-path models/confidence_detector_v1
```

---

## Phase 4 — Evaluation

### 4.1 Core metrics

```python
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Primary metric: macro F1 (penalizes imbalance equally across classes)
# Target: macro F1 > 0.72 on held-out test set
# Baseline (majority class): ~0.33

print(classification_report(y_true, y_pred,
      target_names=["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]))
```

| Metric | Target | Notes |
|---|---|---|
| Macro F1 | > 0.72 | Main metric |
| OVERCLAIMING F1 | > 0.75 | Most practically important class |
| UNDERCLAIMING F1 | > 0.60 | Hardest class, less data |
| CALIBRATED F1 | > 0.75 | Usually easiest |

### 4.2 Calibration of the model itself

Ironically, your confidence detector should itself be well-calibrated. Measure Expected Calibration Error (ECE):

```python
# Check if predicted probabilities match actual accuracy
# A model that says 0.9 confidence should be right ~90% of the time
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true)
plt.plot([0,1],[0,1], linestyle='--')  # perfect calibration line
```

### 4.3 Domain generalization test

Test explicitly on domains not in your training data:
- Train on science + politics → test on finance/legal text
- A drop of more than 15 F1 points signals overfitting to surface cues

### 4.4 Adversarial examples

Manually craft hard cases and track accuracy separately:

```python
HARD_CASES = [
    # Certain phrasing but claim is true and well-supported
    ("The Earth orbits the Sun.", "CALIBRATED"),
    # Hedged phrasing but the underlying claim is still wrong
    ("Some researchers suggest vaccines might cause autism.", "OVERCLAIMING"),
    # Sarcastic overclaiming
    ("Obviously, coffee definitely cures all diseases.", "OVERCLAIMING"),
]
```

---

## Phase 5 — Inference pipeline

### 5.1 Basic inference

```python
from mlx_lm import load, generate

model, tokenizer = load("models/confidence_detector_v1")

def classify(text: str) -> dict:
    prompt = f"""Classify the confidence calibration of this text. 
Reply with: OVERCLAIMING, UNDERCLAIMING, or CALIBRATED, followed by a one-sentence explanation.

Text: {text}"""

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    
    response = generate(
        model, tokenizer,
        prompt=formatted,
        max_tokens=100,
        temp=0.0  # deterministic for classification
    )
    
    # Parse label from response
    for label in ["OVERCLAIMING", "UNDERCLAIMING", "CALIBRATED"]:
        if label in response.upper():
            return {"label": label, "explanation": response}
    
    return {"label": "UNKNOWN", "explanation": response}
```

### 5.2 Batch inference

```python
# For evaluating large datasets efficiently
from mlx_lm import load, generate
import json

def batch_classify(texts: list[str], batch_size: int = 8) -> list[dict]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            results.append(classify(text))
    return results
```

---

## Phase 6 — Iteration roadmap

| Version | What changes | Expected improvement |
|---|---|---|
| v1 | Baseline LoRA, rank 8, 2K iters | Macro F1 ~0.65–0.70 |
| v2 | Add synthetic data, rank 16 | +3–5 F1 points |
| v3 | Add span detection head | New capability: highlight miscalibrated phrase |
| v4 | Domain-specific fine-tune (science / health / finance separately) | +5 F1 on target domain |
| v5 | Distill v4 back into 1B with v4 as teacher | Smaller, faster, same quality |

---

## Estimated timeline

| Phase | Time |
|---|---|
| Dataset collection + labeling | 1–2 weeks |
| Synthetic augmentation | 2–3 days |
| First training run (v1) | 1 day |
| Evaluation + error analysis | 2–3 days |
| Iteration to v2 | 3–5 days |
| Total to working v2 model | ~3–4 weeks |

---

## Quick start checklist

- [ ] `pip install mlx-lm datasets scikit-learn`
- [ ] Download LIAR, SciFact, AVeriTeC from HuggingFace
- [ ] Run label remapping script
- [ ] Generate 3K synthetic triples via API
- [ ] Format as JSONL, split train/val/test
- [ ] Pull `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- [ ] Run first training with rank 8, 2K iters
- [ ] Evaluate on test set, check per-class F1
- [ ] Analyze errors manually (first 50 mistakes)
- [ ] Adjust data balance or LoRA rank and retrain