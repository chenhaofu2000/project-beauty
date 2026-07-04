# project_beauty · Personal Aesthetic Profiler

Extracts an interpretable **aesthetic profile** — subject preferences, tonal tendencies, and mood composition — from a set of user-uploaded images, and generates a readable taste report.

> Built for users with a distinctive but non-professional eye for aesthetics: decomposing hard-to-articulate "taste" into quantifiable, explainable dimensions.

## Design Philosophy

Aesthetics resist direct quantification, but they can be **decomposed by layer**. This project deliberately separates three levels, each handled by a different method:

| Layer | Content | Method | Needs labels? |
|-------|---------|--------|---------------|
| 1 | Objective visual dimensions (lightness / saturation / warmth / contrast / colorfulness) | Pixel statistics | No |
| 2 | Mood & atmosphere (melancholic, warm-dim, serene…) | Rule-based mapping | No |
| 3 | Subjective taste (refinement, niche-ness…) | Preference learning | Yes (planned) |

This layering lets the core product (subject + tone + mood) **run without any labeled data**, reserving costly human annotation only for the most subjective third layer.

## Two-Leg Architecture

The system describes each image with two independent "legs," each doing what it does best:

- **CLIP semantic leg** — classifies **subject** (portrait / landscape / cartoon / illustration / architecture / still-life / food / animal). Semantic recognition is CLIP's native strength, yielding high accuracy.
- **Pixel-statistics leg** — computes **tone** (lightness / saturation / warmth / contrast / colorfulness). These physical properties are more accurate, faster, and more explainable via pixel statistics than via a large model — and they are **subject-invariant** (a dark landscape and a dark portrait share the same lightness value).

Key insight: raw CLIP similarity is dominated by **subject matter** and fails to capture cross-subject tonal consistency; subject-invariant pixel dimensions fill exactly that gap.

## Engineering Decision Log

The project advanced through a loop of *hypothesize → validate with data → keep what works, cut what doesn't*, rather than guesswork:

- ❌ **LAION aesthetic predictor** — validated via Spearman correlation; its "mainstream taste" diverged from (even inverted against) the target user's more niche/refined sensibility, scoring a game screenshot as high-quality. **Cut.**
- ❌ **CLIP mood projection** — measured mood readings clustered too tightly to separate. **Dropped.**
- ✅ **Pixel objective dimensions** — supported by data (weight search converged onto the lightness dimension). **Kept as core.**
- ✅ **CLIP subject classification** — ~90% accuracy on a hand-labeled set. **Kept.**

## Project Structure

```
project_beauty/
├── src/aesthetic/
│   ├── engine.py       # CLIP engine (singleton, centralized model loading)
│   ├── dimensions.py   # Layer 1: objective visual dimensions
│   ├── mood.py         # Layer 2: mood mapping rules
│   ├── subject.py      # CLIP subject-classification leg
│   └── report.py       # Analysis orchestration + report generation
├── scripts/
│   └── analyze.py      # CLI entry point
├── tests/
│   └── test_dimensions.py
└── README.md
```

## Quick Start

```bash
# Install dependencies (requires Python 3.12)
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install open_clip_torch pillow numpy loguru

# Analyze a folder of images
uv run python scripts/analyze.py path/to/images
```

## Sample Output

```
=== Subject Distribution ===
  Portrait      50%
  Illustration  20%
  Animal        20%
  Still-life    10%

=== Mood Distribution ===
  Warm      40%
  Serene    30%
  Warm-dim  20%
  Cool      10%

── Report Draft ──
  You gravitate toward Portrait subjects (50%),
  with a taste composed of 40% Warm + 30% Serene + 20% Warm-dim,
  overall tone: dark-leaning / low-saturation refinement / warm.
```

## Tech Stack

- **CLIP** (open_clip, ViT-B-32 / laion2b) — image semantic encoding & subject classification
- **NumPy / Pillow** — pixel-level dimension computation
- **loguru** — structured logging
- **PyTorch** (CUDA) — model inference

## Roadmap

- [x] Objective dimension extraction (Layer 1)
- [x] Mood mapping (Layer 2)
- [x] CLIP subject classification
- [x] Text report generation
- [ ] Subjective taste learning (Layer 3, via pairwise preference labeling + weight search)
- [ ] Shareable visual report card
- [ ] LLM-polished natural-language report
- [ ] Web interface / cold-start pairwise interaction
```