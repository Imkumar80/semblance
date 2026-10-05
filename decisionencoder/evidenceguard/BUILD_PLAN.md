# EvidenceGuard build plan

EvidenceGuard is the system-track evidence verifier for DecisionEncoder. It follows the
repository principle: **measure first, build second, claim last**.

## Scope and labels

The shipped score is binary: `SUPPORTED = 1`, `UNSUPPORTED = 0`. Normalized datasets also
preserve `fine_label` so future three-class experiments remain possible. Calibrated
abstention is a threshold decision, not a training label.

## Pre-registration

- **Primary endpoint:** binary accuracy on a frozen, hand-checked multi-hop evaluation set,
  scored with every retrieved chunk jointly in one context.
- **Primary comparison:** joint-context scoring versus per-chunk scoring with maximum support
  probability pooling. Both methods use the same checkpoint, examples, token budget, and
  development-selected thresholds.
- **Baselines:** HHEM-2.1, LettuceDetect (any flagged token means unsupported), MiniCheck,
  and DeBERTa NLI, mapped as entailment = supported and contradiction/neutral = unsupported.
  Select thresholds on LLM-AggreFact development data only; report LLM-AggreFact test and
  held-out RAGTruth test. Always report LLM-AggreFact both in full and without
  RAGTruth-derived items.
- **Success rule:** before opening the multi-hop test, joint context must exceed per-chunk
  max pooling by at least **5 percentage points absolute accuracy**, with a paired 95%
  bootstrap confidence interval for the improvement that excludes zero. Otherwise publish
  the result as negative.

## Sequencing

1. Audit dataset licenses before fetching data.
2. Verify source schemas and run every adapter on a tiny local fixture.
3. Produce a data-quality and contamination report before writing training code.
4. Freeze the multi-hop evaluation set and its contamination manifest.
5. Train, calibrate on validation data only, then evaluate on locked test data.
6. Export and measure only after the model evaluation is reproducible.

## Guardrails

- Do not modify `decisionflip/`.
- Use official dataset splits; never train on RAGTruth test data.
- Fail the build if a training ID, normalized claim, or source overlaps an evaluation set.
- Do not record metrics or latency figures until the committed producing script has run.
- Obtain approval before any download larger than 2 GB.
- The current hard-negative perturbation rules are English-only. Direction terms and month
  names live in `data/lexicon_en.json`; other languages require their own reviewed lexicons
  and language-specific validation.
- Entity flips use spaCy NER when available. The dependency-free capitalized-phrase fallback
  is lower precision and labels its output `UNKNOWN` for separate quality review.
- `data/build_train_manifest.py` can compose local normalized JSONL inputs at 30/35/35.
  It requires validation/test manifests for ID, claim, and source overlap checks, samples
  only `split=train` rows, and requires explicit confirmation that RAGTruth-derived inputs
  are cleared for the intended use.
- `data/prepare_ragtruth.py` preserves RAGTruth's official train/test split, filters records
  marked with response-quality issues, maps annotated hallucinations to unsupported rows,
  and removes training rows whose source group appears in the held-out split. Its output
  retains source IDs and annotation spans for audit.
- `data/prepare_train_component.py` extracts train rows from any normalized component and
  removes rows whose IDs, normalized claims, or sources overlap provided evaluation
  manifests. The manifest composer still fails on any remaining overlap.
- `decisionencoder/modeling.py` and `decisionencoder/train.py` implement the ModernBERT
  binary cross-encoder and a manifest-only trainer. The trainer dynamically pads batches,
  uses SDPA and a 10% warmup schedule, and rejects non-train manifest rows.
- `data/synthesize_vitaminc_multichunk.py` creates paired, same-page two-chunk examples
  from supported VitaminC train facts. Its generated claims and missing-evidence negatives
  are synthetic and require semantic review; it is augmentation, not a frozen benchmark.

## Current local training artifacts

The current pipeline can create a roughly 42K-row exact 30/35/35 training manifest when
the local sources are available. Remove overlap against the validation/test manifests
before composing:

```powershell
python .\decisionencoder\evidenceguard\data\prepare_train_component.py `
  --input .\decisionencoder\evidenceguard\data\hard_negatives.with_entities.jsonl `
  --eval-manifest .\decisionencoder\evidenceguard\data\hard_negatives.with_entities.jsonl `
    .\decisionencoder\evidenceguard\data\ragtruth_heldout_test.jsonl `
    .\decisionencoder\evidenceguard\data\vitaminc_norm.jsonl `
    .\decisionencoder\evidenceguard\data\synthetic_multichunk.jsonl `
  --out .\decisionencoder\evidenceguard\data\hard_negatives_train.jsonl --force

python .\decisionencoder\evidenceguard\data\prepare_train_component.py `
  --input .\decisionencoder\evidenceguard\data\ragtruth_train.jsonl `
  --eval-manifest .\decisionencoder\evidenceguard\data\hard_negatives.with_entities.jsonl `
    .\decisionencoder\evidenceguard\data\ragtruth_heldout_test.jsonl `
    .\decisionencoder\evidenceguard\data\vitaminc_norm.jsonl `
    .\decisionencoder\evidenceguard\data\synthetic_multichunk.jsonl `
  --out .\decisionencoder\evidenceguard\data\ragtruth_train_clean.jsonl --force

python .\decisionencoder\evidenceguard\data\synthesize_vitaminc_multichunk.py `
  --input .\decisionencoder\evidenceguard\data\vitaminc_norm.jsonl `
  --out .\decisionencoder\evidenceguard\data\vitaminc_multichunk_train.jsonl `
  --report .\decisionencoder\evidenceguard\data\vitaminc_multichunk_report.json `
  --max-rows 15000 --seed 0 --force

python .\decisionencoder\evidenceguard\data\prepare_train_component.py `
  --input .\decisionencoder\evidenceguard\data\vitaminc_multichunk_train.jsonl `
  --eval-manifest .\decisionencoder\evidenceguard\data\hard_negatives.with_entities.jsonl `
    .\decisionencoder\evidenceguard\data\ragtruth_heldout_test.jsonl `
    .\decisionencoder\evidenceguard\data\vitaminc_norm.jsonl `
    .\decisionencoder\evidenceguard\data\synthetic_multichunk.jsonl `
  --out .\decisionencoder\evidenceguard\data\vitaminc_multichunk_train_clean.jsonl --force

python .\decisionencoder\evidenceguard\data\build_train_manifest.py `
  --hard-negatives .\decisionencoder\evidenceguard\data\hard_negatives_train.jsonl `
  --multichunk .\decisionencoder\evidenceguard\data\vitaminc_multichunk_train_clean.jsonl `
  --ragtruth-train .\decisionencoder\evidenceguard\data\ragtruth_train_clean.jsonl `
  --eval-manifest .\decisionencoder\evidenceguard\data\hard_negatives.with_entities.jsonl `
    .\decisionencoder\evidenceguard\data\ragtruth_heldout_test.jsonl `
    .\decisionencoder\evidenceguard\data\vitaminc_norm.jsonl `
    .\decisionencoder\evidenceguard\data\synthetic_multichunk.jsonl `
  --out .\decisionencoder\evidenceguard\data\train_manifest.jsonl `
  --confirm-ragtruth-rights --force
```

The manifest currently contains 42,380 rows. It is a training artifact, not an evaluation
set; the generated VitaminC multi-chunk examples are weakly supervised and require audit.
Do not interpret a decreasing smoke-test training loss as evidence of generalization.
The 30/35/35 shares are currently fixed in the composer; it intentionally has no `--ratio`
option. Exact component counts require a total row count divisible by 20.
