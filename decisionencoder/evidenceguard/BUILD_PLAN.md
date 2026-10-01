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
