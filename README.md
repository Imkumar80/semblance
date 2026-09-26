# semblance

> DecisionEncoder investigates whether compact language encoders can be trained to distinguish decision-critical functional incompatibilities that remain semantically similar, using DecisionFlip as a diagnostic benchmark and applying the resulting capability to tool routing, evidence verification, policy gating, and calibrated abstention.

**Guiding principle:** Measure first. Build second. Claim last.

## Layout

```
semblance/
├── DecisionEncoder_Master_Research_and_Project_Plan.md   # full research & project plan
├── decisionflip/                                         # research track: the benchmark
│   ├── generate_pilot.py    # rule-based pilot pair generator
│   ├── pilot_pairs.json     # generated pilot set (committed for reproducibility)
│   └── eval_harness.py      # Section 16 critical-first-experiment harness
└── decisionencoder/                                       # system track: the model (empty for now)
```

## Status

Currently in **Phase 2 — DecisionFlip pilot** (see the plan doc, Section 28). No model training has started.

## Quickstart

```bash
cd decisionflip
python generate_pilot.py --n-per-category 7 --out pilot_pairs.json

pip install sentence-transformers --break-system-packages
python eval_harness.py --pairs pilot_pairs.json --model all-MiniLM-L6-v2
```

The eval harness requires network access to huggingface.co to download model weights — run it locally, not in a restricted sandbox.

## Read first

`DecisionEncoder_Master_Research_and_Project_Plan.md` — Section 16 (Critical First Experiment) and Section 38 (Guiding Principle) explain what "done" looks like before any training happens.
