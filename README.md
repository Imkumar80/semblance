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
└── decisionencoder/                                       # system track: EvidenceGuard model and data pipeline
    ├── modeling.py          # ModernBERT binary groundedness classifier
    ├── train.py             # JSONL manifest trainer (run as a module)
    └── evidenceguard/data/  # normalization, synthesis, and leakage-checked manifests
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

## DecisionFlip v3 experiment

The v2 generator writes explicit template/policy IDs and deterministic group-level train/dev/test assignments. The 1,000-example setting applies per category where combinations exist; finite categories can produce fewer examples. Scope and context-policy evaluation reports within-family, held-out-template, held-out-policy, and locked-policy-test results. Numerical results are calibration-only.

```powershell
cd .\decisionflip
python .\generate_pilot_v2.py --n-per-category 1000 --out .\pilot_v3_large.json
python .\eval_v2.py --data .\pilot_v3_large.json --model BAAI/bge-small-en-v1.5 --out .\results_bge_small_v3.json
python .\eval_v2.py --data .\pilot_v3_large.json --model BAAI/bge-large-en-v1.5 --out .\results_bge_large_v3.json
```

TF-IDF/logistic regression is included as a lexical diagnostic. To additionally run Laya, install its package and pass either `--laya-checkpoint base` or `--laya-checkpoint typed-decisions`; this downloads the selected checkpoint. To test another LLM, export only the locked context-policy test cases, have the model return the requested `record_id`/label JSON, then evaluate those predictions:

```powershell
python .\eval_v2.py --data .\pilot_v3_large.json --export-llm-cases .\context_policy_llm_cases.json
python .\eval_v2.py --data .\pilot_v3_large.json --model hashing --llm-predictions .\llm_predictions.json --out .\results_llm.json
```

The family/policy bootstrap intervals reflect the small number of independent groups; they are not substitutes for a larger human-reviewed test set. The Phase 6 success threshold and matched plain-fine-tuning comparator are preregistered in the research plan's Section 18.

## EvidenceGuard training

EvidenceGuard data preparation and model training live under `decisionencoder/`. The
VitaminC multi-chunk generator creates synthetic paired two-chunk examples; review its
report and `review_note` fields before treating those examples as benchmark-quality labels.
The manifest composer samples exact component proportions without replacement and rejects
train/evaluation overlap.

From the repository root, a bounded CPU smoke test can be run with:

```powershell
python -m decisionencoder.train `
  --manifest .\decisionencoder\evidenceguard\data\train_manifest.jsonl `
  --out .\decisionencoder\models\evidenceguard_smoke `
  --epochs 1 --batch-size 2 --max-length 128 --limit-train-rows 4 --device cpu
```

Use a CUDA-enabled environment for full training; `--device auto` selects CUDA when
available. See `decisionencoder/evidenceguard/BUILD_PLAN.md` for data provenance, split,
and licensing safeguards.

## Read first

`DecisionEncoder_Master_Research_and_Project_Plan.md` — Section 16 (Critical First Experiment) and Section 38 (Guiding Principle) explain what "done" looks like before any training happens.
