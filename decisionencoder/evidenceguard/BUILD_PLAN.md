# EvidenceGuard build plan

EvidenceGuard is the system-track evidence verifier for DecisionEncoder. It follows the
repository principle: **measure first, build second, claim last**.

## Scope and labels

The shipped score is binary: `SUPPORTED = 1`, `UNSUPPORTED = 0`. Normalized datasets also
preserve `fine_label` so future three-class experiments remain possible. Calibrated
abstention is a threshold decision, not a training label.

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
