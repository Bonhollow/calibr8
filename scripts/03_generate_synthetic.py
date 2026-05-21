#!/usr/bin/env python3
"""
Step 3: Generate synthetic training data via Claude API or fallback to hand-crafted examples.

For each calibrated sentence, generates overclaiming + underclaiming variants.
Requires ANTHROPIC_API_KEY env var. Falls back to built-in examples if unavailable.
"""

import json
import os
import time
from pathlib import Path

REMAPPED_DIR = Path(__file__).parent.parent / "data" / "remapped"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
STATE_FILE = SYNTHETIC_DIR / ".progress.json"

AUGMENTATION_PROMPT = """Given this calibrated sentence, rewrite it in two ways:
1. Overclaiming version — add unwarranted certainty (use words like "proven", "definitely", "always", "undeniably")
2. Underclaiming version — add excessive hedging (use words like "might possibly", "some suggest", "could perhaps")

Keep the core topic the same. Make the changes sound natural, not cartoonish.

Sentence: {sentence}

Return ONLY valid JSON with no other text:
{{"overclaiming": "...", "underclaiming": "...", "original": "..."}}"""

# Hand-crafted fallback examples across domains
FALLBACK_TRIPLES = [
    # Science
    {"original": "A meta-analysis of 12 studies found a statistically significant correlation between sleep duration and cognitive performance.",
     "overclaiming": "Science has proven that sleep definitively determines your cognitive abilities — always.",
     "underclaiming": "Some researchers think there might possibly be some kind of link between sleep and how people think, though it's unclear."},
    {"original": "The vaccine reduced symptomatic infection by 67% in the trial population.",
     "overclaiming": "The vaccine is proven to completely prevent infection in everyone who takes it.",
     "underclaiming": "It is conceivable that the vaccine could perhaps have some minor effect on infection rates in certain people."},
    {"original": "Global average temperatures have risen by approximately 1.1°C since pre-industrial times.",
     "overclaiming": "Temperatures have undeniably skyrocketed and will definitely destroy all life on Earth.",
     "underclaiming": "Some data might possibly suggest temperatures could perhaps be slightly different than before."},
    # Health
    {"original": "Regular aerobic exercise has been associated with reduced risk of cardiovascular disease in multiple cohort studies.",
     "overclaiming": "Exercise is a guaranteed cure for heart disease — science has shown this beyond any doubt.",
     "underclaiming": "Some people believe exercise might possibly have some unclear connection to heart health."},
    {"original": "The Mediterranean diet was linked to a 25% lower risk of type 2 diabetes in a 10-year longitudinal study.",
     "overclaiming": "The Mediterranean diet has been proven to completely prevent diabetes — it's an irrefutable fact.",
     "underclaiming": "It is conceivable that some dietary patterns could perhaps have a tentative relationship with certain metabolic conditions."},
    {"original": "Excessive sugar consumption is associated with increased risk of dental caries according to WHO systematic reviews.",
     "overclaiming": "Sugar always causes cavities — this is a proven, undeniable fact that no one can dispute.",
     "underclaiming": "Some researchers think sugar might possibly have some unclear connection to dental issues."},
    # Politics / Policy
    {"original": "Countries with higher minimum wages tend to have lower poverty rates, though confounding factors exist.",
     "overclaiming": "Higher minimum wages always eliminate poverty — economics has conclusively proven this.",
     "underclaiming": "It is conceivable that wage policies could perhaps have some minor, unclear effect on poverty."},
    {"original": "Voter ID laws have been shown to reduce turnout among minority populations by 2-3 percentage points in several studies.",
     "overclaiming": "Voter ID laws are definitively proven to be racist tools that always suppress minority votes completely.",
     "underclaiming": "Some suggest voter ID might possibly affect some voters in some unclear way, though it's uncertain."},
    # Finance
    {"original": "Index funds have outperformed the majority of actively managed funds over 15-year periods according to SPIVA data.",
     "overclaiming": "Index funds are guaranteed to always beat every active manager — this is an irrefutable financial fact.",
     "underclaiming": "It is conceivable that some passive investment strategies could perhaps perform somewhat differently than others."},
    {"original": "Higher interest rates typically slow economic growth by increasing borrowing costs for businesses and consumers.",
     "overclaiming": "Interest rate hikes always and definitively crash the economy — this has been proven without exception.",
     "underclaiming": "Some economists suggest that interest rates might possibly have some tentative connection to economic activity."},
    # Technology
    {"original": "Large language models can generate fluent text but frequently produce factual errors, known as hallucinations.",
     "overclaiming": "AI is proven to be completely unreliable and will never be able to produce accurate information.",
     "underclaiming": "Some researchers think AI text tools might possibly sometimes produce outputs that could perhaps contain inaccuracies."},
    {"original": "Screen time exceeding 4 hours daily was correlated with poorer sleep quality in adolescents in a 2023 cohort study.",
     "overclaiming": "Screens are definitively proven to destroy children's sleep — this is an undeniable fact confirmed by all research.",
     "underclaiming": "It is conceivable that screen use could perhaps have some minor, unclear relationship with sleep patterns in some young people."},
    # General knowledge
    {"original": "Bilingual individuals show delayed onset of dementia symptoms by approximately 4-5 years in observational studies.",
     "overclaiming": "Speaking two languages is a guaranteed, proven cure for dementia — science has shown this conclusively.",
     "underclaiming": "Some suggest that knowing another language might possibly have some tentative connection to brain health."},
    {"original": "Deforestation in the Amazon has accelerated, with satellite data showing a 22% increase in 2023 compared to the prior decade average.",
     "overclaiming": "The Amazon is definitively and completely destroyed — deforestation has proven to be totally irreversible.",
     "underclaiming": "Some satellite data might possibly suggest that forest coverage could perhaps be changing in some regions."},
    {"original": "Microplastics have been detected in human blood samples across multiple independent studies.",
     "overclaiming": "Microplastics are proven to be killing everyone — they are undeniably the greatest health threat ever.",
     "underclaiming": "It is conceivable that some particles could perhaps be present in some biological samples, though it's unclear."},
]


def generate_with_claude(sentences: list[str], max_count: int = 500) -> list[dict]:
    """Generate synthetic triples using Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"  [WARN] Cannot use Claude API: {e}")
        return []

    # Load progress state
    done = set()
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            done = set(json.load(f).get("done", []))

    results = []
    for i, sentence in enumerate(sentences[:max_count]):
        if sentence in done:
            continue
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": AUGMENTATION_PROMPT.format(sentence=sentence)}],
            )
            text = response.content[0].text.strip()
            triple = json.loads(text)
            if "overclaiming" in triple and "underclaiming" in triple:
                results.append(triple)
                done.add(sentence)
                if len(results) % 25 == 0:
                    print(f"    Generated {len(results)}/{max_count}...")
                    # Save progress
                    with open(STATE_FILE, "w") as f:
                        json.dump({"done": list(done)}, f)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"    [WARN] Failed on item {i}: {e}")
            time.sleep(2)

    # Final save
    with open(STATE_FILE, "w") as f:
        json.dump({"done": list(done)}, f)

    return results


def triples_to_records(triples: list[dict]) -> list[dict]:
    """Convert triples to labeled records."""
    records = []
    for t in triples:
        if t.get("overclaiming"):
            records.append({"text": t["overclaiming"], "label": 0, "label_name": "OVERCLAIMING", "source": "synthetic"})
        if t.get("underclaiming"):
            records.append({"text": t["underclaiming"], "label": 1, "label_name": "UNDERCLAIMING", "source": "synthetic"})
        if t.get("original"):
            records.append({"text": t["original"], "label": 2, "label_name": "CALIBRATED", "source": "synthetic"})
    return records


def main():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "synthetic.jsonl"

    print("=" * 60)
    print("STEP 3: Generating synthetic training data")
    print("=" * 60)

    # Collect calibrated sentences from remapped data as seeds
    seeds = []
    for f in sorted(REMAPPED_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                if row["label"] == 2 and len(row["text"]) > 30:
                    seeds.append(row["text"])

    print(f"  Found {len(seeds)} calibrated seed sentences")

    # Try Claude API first
    api_triples = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  Using Claude API for generation...")
        api_triples = generate_with_claude(seeds, max_count=500)
        print(f"  Generated {len(api_triples)} triples via API")
    else:
        print("  [INFO] No ANTHROPIC_API_KEY found — using fallback examples only")

    # Combine API + fallback
    all_triples = api_triples + FALLBACK_TRIPLES
    records = triples_to_records(all_triples)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    label_dist = {0: 0, 1: 0, 2: 0}
    for r in records:
        label_dist[r["label"]] += 1

    print(f"\n  Synthetic records: {len(records)}")
    print(f"    OC: {label_dist[0]}  |  UC: {label_dist[1]}  |  CAL: {label_dist[2]}")
    print(f"  Saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
