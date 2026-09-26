# DecisionEncoder — Master Research & Project Plan

## 1. Executive Summary

DecisionEncoder is a research and engineering project investigating whether a compact, non-generative language encoder can make routine agent decisions more reliably than ordinary semantic-similarity systems.

The project has two connected tracks:

1. **Research track — DecisionFlip**
   - Identify and measure cases where two inputs are semantically very similar but require different decisions.
   - Build a controlled benchmark around these cases.
   - Test existing embedding models, rerankers, and other decision systems.
   - Determine whether decision-critical hard-negative training improves performance.

2. **System track — DecisionEncoder**
   - If the benchmark reveals a real weakness, develop a compact encoder specialized for functional compatibility, uncertainty, and abstention.
   - Apply it to ToolRouter, EvidenceGuard, and PolicyGate.
   - Build an efficient retrieval → reranking → decision runtime.

The project is deliberately hypothesis-driven. We will not assume that existing models fail, and we will not assume that a new architecture is necessary. The first scientific objective is to measure the problem.

---

## 2. The Core Idea in Plain English

Ordinary embedding models are very good at answering:

> “Are these two pieces of text similar?”

DecisionEncoder is interested in a different question:

> “Should these two pieces of text lead to the same action?”

Those questions are related, but they are not identical.

Example:

- “Transfer $999 to Alice.”
- “Transfer $1,000 to Alice.”

A semantic model may consider these sentences extremely similar.

But suppose the policy is:

- below $1,000 → ALLOW
- $1,000 or more → REQUIRE APPROVAL

Then a tiny change produces a different decision.

That is a **DecisionFlip**.

The research asks whether current language representations reliably preserve these decision-critical distinctions.

**Caveat on the flagship example.** A pure numerical threshold ($999 vs $1,000) is close to a structured-extraction problem — a reasonable production system would parse the amount and apply the rule directly, not rely on embedding similarity. This example is useful as the *simplest possible diagnostic* because ground truth is trivial to define, but it is not the strongest motivating case for why embeddings specifically should matter. The categories that matter most for the paper's motivation are the ones where regex/extraction is not a viable shortcut — **scope** (staging vs. production), **permission** (authorized vs. unauthorized), and **action** (create vs. delete) — because there the correct decision depends on context that isn't a single extractable token. The numerical case should be framed explicitly as the calibration/sanity-check category, not the headline motivating example.

---

## 3. Research Hypothesis

### Primary hypothesis

Semantic embedding models may perform strongly on ordinary semantic similarity while being less reliable when a small textual change changes the correct action.

### Secondary hypothesis

Training with decision-critical hard negatives may improve this capability without requiring a large generative model.

These are hypotheses, not established facts. The experiments must determine whether they are true.

---

## 4. DecisionFlip Benchmark

A DecisionFlip is a pair of closely related inputs:

- highly semantically similar,
- changed by a small or controlled modification,
- associated with different correct decisions.

Example:

**Rule:** amount < $1,000 → ALLOW; amount >= $1,000 → REQUIRE_APPROVAL.

**A:** “Transfer $999 to Alice.” → ALLOW

**B:** “Transfer $1,000 to Alice.” → REQUIRE_APPROVAL

The benchmark is designed to make this situation measurable.

---

## 5. Why We Need Our Own Benchmark

Existing datasets are useful, but they were generally created for different objectives such as semantic similarity, retrieval, entailment, tool selection, or classification.

DecisionFlip is specifically designed to test:

> Can a model recognize a small change that should cause a different decision?

Therefore the core benchmark will be controlled rather than simply being a random collection of existing examples.

Existing datasets will still be used for transfer and real-world evaluation.

---

## 6. Dataset Strategy

We will use three complementary data sources.

### A. Controlled DecisionFlip data

The main benchmark.

We define explicit rules and generate examples around those rules. Labels come from the rules.

### B. Existing datasets

The original DecisionEncoder plan identifies ToolBench/function-calling traces for ToolRouter, FEVER/MNLI/ANLI/RAGTruth-style data for EvidenceGuard, and synthetic policy data for PolicyGate.

### C. Realistic agent examples

Construct/adapt realistic examples around tool selection, policy decisions, and evidence verification.

---

## 7. Controlled Data Generation

We should not manually write thousands of examples.

Instead, define structured variables.

Example template:

> Transfer {amount} to {recipient}.

Variables can include:

- amounts,
- recipients,
- wording,
- contexts,
- environments,
- actions.

The program generates combinations and calculates the correct label from the rule.

---

## 8. DecisionFlip Categories

Initial categories:

### Numerical thresholds
Example: $999 → ALLOW; $1,000 → APPROVAL.

### Scope
Examples: staging → production; internal → external; sandbox → live.

### Permission
Examples: authorized → unauthorized; approved → unapproved.

### Entity
Only when the entity affects the decision.

### Action
Examples: create → delete; enable → disable.

### Tool function
Examples: create_event vs list_events.

### Negation
Examples: allow vs do not allow.

Negation and tool-function mapping should not automatically be claimed as novel; they have related prior work.

---

## 9. Ground Truth

Ground truth should come from explicit rules wherever possible.

We should not ask an LLM to decide whether the policy label is correct.

Instead, automated checks can verify whether generated text faithfully expresses the intended variables.

For example:

- Did the sentence state $1,000?
- Did it refer to the intended account?
- Did it preserve the intended environment?

The uploaded benchmark review specifically recommends rule-defined ground truth and model-based faithfulness checking rather than model-based correctness judging.

---

## 10. Dataset Quality Control

Quality control will have several layers:

1. Programmatic checks for values, entities, variables, labels, and rules.
2. Narrow faithfulness checks using open-weight models where useful.
3. Self-consistency to identify unstable checks.
4. Human spot checks for uncertain cases.
5. Contamination checks against large public corpora and known benchmarks.

The goal is to use humans where they add the most value rather than manually labeling thousands of deterministic examples.

**Concrete contamination check plan.** Two risks matter: (a) generated pairs accidentally duplicating patterns already present in the existing datasets used for transfer evaluation (ToolBench, FEVER/MNLI/ANLI), which would inflate transfer numbers; (b) the templates themselves being distinctive enough that a model could learn to detect "this is a synthetic DecisionFlip pair" rather than solving the actual task. Minimum viable check before Phase 5 scale-up:
- n-gram overlap (e.g. 8-gram) between generated sentences and the transfer-eval corpora, flagging any pair above a set overlap threshold for review.
- Embedding-based near-duplicate search (encode both the generated set and a sample of the source corpora, flag nearest neighbors above a similarity threshold).
- A held-out "template family" that is never used during any training or hard-negative mining, reserved purely for the generalization split in Section 12.

---

## 11. Dataset Size

We should not immediately generate tens of thousands of examples.

Development:

20–50 pairs
→ pilot
→ analyze
→ hundreds
→ quality validation
→ thousands

The benchmark proposal discussed a 500–1,000 pair pilot-scale release; this should be treated as an initial release target rather than a permanent maximum.

---

## 12. Generalization Splits

A random split is not enough.

We should test:

- unseen templates,
- unseen entities,
- unseen numerical values,
- unseen domains.

This reduces the chance that models simply memorize benchmark patterns.

---

## 13. Baseline Models

Initial baseline categories:

1. MiniLM-L6
2. BGE-small
3. BGE-large
4. BM25
5. One generative LLM
6. Optionally a calibrated typed-decision classifier such as Laya

The benchmark review recommends keeping the baseline set focused rather than adding models only for breadth.

---

## 14. What We Measure

### Decision Accuracy
Did the model make the correct decision?

### Flip Consistency
When the correct decision changes, does the model also change its decision correctly?

### Similarity Gap
How similar are the two inputs?

The combination of Flip Consistency and Similarity Gap is central to the benchmark design.

**Definition and circularity risk.** "High similarity" must be defined before pairs are selected, not derived from the same model being evaluated -- otherwise the benchmark is circular (e.g. using MiniLM's own similarity score to decide which pairs count as "hard," then reporting MiniLM fails on them). Two workable options:

1. **Model-agnostic construction**: pairs are hard by *construction* (minimal edit distance / single-variable change under a fixed template), and similarity is only *measured*, never used to filter, per model under test.
2. **Reference-model similarity**: define "high similarity" using one fixed reference embedding model (declared upfront, not in the baseline comparison set), so the threshold is independent of any model being scored.

Report Similarity Gap as an absolute cosine value per pair alongside a percentile within the corpus, so results aren't sensitive to a single arbitrary cutoff.

### Hard-negative performance
Can the model distinguish near-correct but functionally wrong candidates?

### Calibration
Potential metrics:

- ECE,
- reliability diagrams,
- risk-coverage,
- selective accuracy,
- abstention rate.

### Retrieval
For ToolRouter:

- Recall@1,
- Recall@5,
- Recall@10,
- Recall@50.

### Systems
- latency,
- throughput,
- memory,
- candidate scaling,
- CPU/GPU requirements.

---

## 15. Representation Analysis

For each DecisionFlip pair, calculate embedding similarity.

Then investigate cases where:

> high semantic similarity + different correct decision

This helps determine whether ordinary embedding geometry preserves decision-critical information.

Visualization methods such as PCA/UMAP can be used as diagnostics, not as the primary evidence.

---

## 16. Critical First Experiment

Before training anything new:

1. Create approximately 20–50 DecisionFlip pairs.
2. Run MiniLM.
3. Measure similarity.
4. Test decision discrimination.
5. Inspect failures.
6. Determine whether a measurable problem exists.

Possible outcomes:

### A — Existing models already work well
The DecisionFlip hypothesis is weak; reconsider.

### B — Failure in specific categories
We have a concrete research problem.

### C — Broad failure
The case for a decision-focused model becomes stronger.

---

## 17. Decision-Critical Hard Negatives

A major proposed intervention is to train with negatives that are:

- semantically close,
- same domain,
- similar wording,
- functionally incompatible.

Example:

Query:
> Create a calendar event.

Correct:
> calendar.create_event

Hard negative:
> calendar.list_events

This is more informative than an easy negative such as “Bake a chocolate cake.”

---

## 18. Decision-Focused Training

Ordinary semantic training roughly encourages:

similar meaning → close vectors

different meaning → distant vectors

DecisionEncoder investigates an additional signal:

same semantic domain + different required action → separate functionally.

Possible objectives include:

- contrastive loss,
- triplet loss,
- multiple-negative ranking,
- classification loss,
- combinations.

We should experiment rather than commit in advance.

**Starting point and stopping rule.** To avoid open-ended tinkering once compute is being spent:
- Default starting loss: **Multiple Negatives Ranking (in-batch negatives + explicit decision-critical hard negatives injected per batch)** — it is the standard, well-understood starting point in the sentence-transformers ecosystem and needs the least hyperparameter search (no margin to tune, unlike triplet loss).
- Fixed run budget for the first pass: no more than 3-4 training configurations (e.g. MNR baseline, MNR + hard negatives, triplet with 1-2 margins) before moving to failure analysis, rather than an open-ended sweep.
- Decision rule for moving on: if none of the first-pass configurations improve Flip Consistency over the best baseline by a pre-registered minimum margin, treat that as a genuine negative result (Section 31) rather than expanding the search — go back to failure analysis instead of adding more loss variants.

---

## 19. DecisionEncoder Architecture

Only after the benchmark demonstrates a need should the architecture be finalized.

Conceptually:

input
→ pretrained language encoder
→ representation
→ compatibility decision + uncertainty/abstention

For candidate-heavy tasks:

input
→ bi-encoder retrieval
→ top-K candidates
→ cross-encoder / decision reranking
→ decision
→ confidence
→ ABSTAIN if uncertain

The original implementation plan proposes retrieval plus reranking for variable-candidate tasks.

---

## 20. ToolRouter

Question:

> Which tool should the agent call?

For thousands of tools:

10,000 candidates
→ cheap embedding retrieval
→ Top 50
→ reranking
→ best candidates
→ decision / abstain

The original plan uses cached candidate embeddings and ANN-style retrieval followed by reranking.

---

## 21. EvidenceGuard

Question:

> Does the evidence support the claim?

Conceptually:

claim + evidence
→ DecisionEncoder
→ support / contradiction / insufficient / abstain

The exact label scheme will follow the selected datasets and experiments.

---

## 22. PolicyGate

Question:

> Should this action be allowed?

Conceptually:

request + context + policy
→ PolicyGate
→ ALLOW / DENY / REQUIRE_APPROVAL / ABSTAIN

This is naturally compatible with explicit DecisionFlip rules.

---

## 23. Abstention

The model should not be forced to choose when uncertain.

Confident
→ decision

Uncertain
→ ABSTAIN
→ LLM / human / clarification

We will measure whether confidence is meaningful through calibration and risk-coverage analysis.

---

## 24. Decision Runtime

The eventual runtime:

request
→ candidate retrieval
→ Top-K
→ decision reranking
→ confidence check
→ action OR abstain

Low latency and predictable resource use are engineering goals.

A sub-20ms target should be treated as a benchmark target under specified hardware/workload conditions, not as a result claimed beforehand.

---

## 25. Ablations

If DecisionEncoder improves performance, we must determine why.

Possible ablations:

- full model,
- remove decision-critical hard negatives,
- remove abstention,
- remove a loss component,
- remove reranking.

The goal is to connect each component to a measurable improvement.

---

## 26. Transfer Experiments

After DecisionFlip, test whether the capability transfers to:

- ToolRouter,
- EvidenceGuard,
- PolicyGate.

This prevents the work from being only a benchmark-specific optimization.

---

## 27. Compute Plan

Available:

- NVIDIA A6000 — 48 GB VRAM
- A100 compute allocation

Use them strategically.

### A6000
Development, local model testing, small/medium training, baseline evaluation.

### A100
Larger inference, large benchmark runs, training experiments, sweeps, final evaluation.

Do not spend large compute before the pilot establishes that the problem is real.

**Explicit sizing.** Phases 0-5 (understanding, pilot, baselines, failure analysis, initial benchmark construction at 20-1,000 pairs) run entirely on CPU or a single A6000, with baseline models (MiniLM, BGE-small/large, BM25) needing only inference, not training. The A100 allocation is not needed until **Phase 6** (decision-focused training) at the earliest, and even then a single A6000 is likely sufficient for a compact encoder — reserve the A100 for Phase 7-11 sweeps, ablations, and large-scale final evaluation once Phase 4 has confirmed the problem is real. Stating this explicitly avoids requesting large-scale compute before there is evidence it is needed.

---

## 28. Research Phases

### Phase 0 — Existing model understanding
Tokenizer, Transformer, pooling, normalization, cosine similarity, embeddings.

### Phase 1 — Training understanding
Positive/negative pairs, losses, gradients, contrastive learning, hard negatives.

### Phase 2 — DecisionFlip pilot
20–50 controlled pairs.

### Phase 3 — Baseline experiment
Existing models.

### Phase 4 — Failure analysis
Similarity, decision errors, categories, representations.

### Phase 5 — Benchmark construction
Scale after validating the phenomenon.

### Phase 6 — Decision-focused training
First DecisionEncoder experiments.

### Phase 7 — Ablations
Identify causal contributions.

### Phase 8 — Generalization
Unseen templates, entities, values, domains.

### Phase 9 — Agent tasks
ToolRouter, EvidenceGuard, PolicyGate.

### Phase 10 — Runtime
Retrieval, reranking, batching, caching, latency.

### Phase 11 — Final evaluation
Accuracy, Flip Consistency, calibration, latency, memory, scaling.

### Phase 12 — Release
Dataset, models, evaluation code, runtime, documentation, paper.

---

## 29. What Makes This Research?

The engineering project is:

> Build a compact decision encoder and runtime.

The research project is:

> Investigate whether semantic representations are insufficient for certain decision-critical distinctions, measure that failure systematically, and test whether decision-focused training improves it.

The benchmark gives us the measurement.

The model gives us the intervention.

The experiments determine whether the intervention works.

---

## 30. What Would Count as a Strong Result?

A strong result would show:

1. Existing models perform well on ordinary semantic tasks.
2. They have measurable weaknesses on DecisionFlip.
3. Failures concentrate in particular decision-critical categories.
4. Errors correlate with high semantic similarity.
5. Decision-critical hard-negative training improves performance.
6. Improvements survive unseen-template/domain tests.
7. Calibration and abstention add useful value.
8. The system remains computationally practical.
9. Ablations identify which components matter.

We should not decide the numerical thresholds before running experiments.

---

## 31. What Would Falsify the Idea?

We should accept negative results.

Examples:

- existing models already solve DecisionFlip reliably,
- hard-negative training gives no improvement,
- gains disappear on unseen templates,
- gains come only from larger models,
- benchmark artifacts explain the result,
- calibration does not improve,
- runtime costs outweigh benefits.

If that happens, we change direction rather than forcing a positive claim.

---

## 32. Paper Directions

There are potentially two connected paper directions.

### Paper A — Benchmark / Dataset Track

Core contribution:

DecisionFlip benchmark + evaluation methodology.

Possible structure:

1. Motivation
2. Problem definition
3. Related benchmark landscape
4. Dataset construction
5. Metrics
6. Baselines
7. Results
8. Failure analysis
9. Generalization
10. Limitations

The current benchmark review considers this a plausible Datasets & Benchmarks contribution if the verification and positioning issues are handled carefully.

### Paper B — Model Track

If experiments show a meaningful improvement:

DecisionEncoder
+ decision-focused training
+ DecisionFlip
+ ToolRouter/EvidenceGuard/PolicyGate

This could become a model/system paper.

We should not decide which paper is stronger until we have results.

---

## 33. Novelty Boundary

We are not claiming to have invented:

- embeddings,
- hard negatives,
- contrastive learning,
- tool retrieval,
- cross-encoders,
- calibration,
- abstention,
- policy classification.

The potential contribution is the combination of a specific diagnostic problem and methodology:

> minimal textual changes that produce different decisions despite high semantic similarity

together with a formal benchmark protocol and, if experiments support it, a decision-focused representation/training method.

The benchmark review also notes that related areas such as negation and tool-function mapping already have substantial prior work, so those should not be presented as wholly new problems.

---

## 34. Reproducibility and Release

Where licensing permits, release:

- benchmark data,
- generation rules/templates,
- generation code,
- validation code,
- evaluation code,
- baseline configurations,
- model checkpoints,
- training configurations,
- runtime implementation,
- benchmark results.

MTEB integration can be submitted for consideration, but it should not be described as accepted until it is actually accepted. A standalone evaluation configuration should remain available.

---

## 35. Complete Project Architecture

DecisionEncoder Project

Research Track
→ DecisionFlip
→ benchmark
→ existing-model evaluation
→ failure analysis
→ decision-focused training

System Track
→ DecisionEncoder
→ ToolRouter
→ EvidenceGuard
→ PolicyGate
→ abstention
→ decision runtime

Both tracks
→ rigorous evaluation
→ reproducible release
→ research paper

---

## 36. Where We Are Now

We are not yet at the model-building stage.

Current learning path:

Sentence Transformers
→ embeddings ✓
→ pooling ✓
→ cosine similarity ✓
→ training and loss ← NEXT
→ contrastive learning
→ hard negatives
→ DecisionFlip pilot
→ baseline experiments
→ DecisionEncoder design

The next educational step is:

> Understand how training changes the embedding space.

We will use a tiny example before touching DecisionEncoder.

---

## 37. One-Sentence Project Definition

> **DecisionEncoder investigates whether compact language encoders can be trained to distinguish decision-critical functional incompatibilities that remain semantically similar, using DecisionFlip as a diagnostic benchmark and applying the resulting capability to tool routing, evidence verification, policy gating, and calibrated abstention.**

---

## 38. Guiding Principle

**Measure first. Build second. Claim last.**

First prove whether the problem exists.

Then determine where and why models fail.

Then try the smallest intervention that could fix it.

Then run ablations and generalization tests.

Only after the evidence supports the result should we claim a research contribution.
