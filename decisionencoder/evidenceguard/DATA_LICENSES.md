# EvidenceGuard data-license audit

Status: initial audit, 2026-10-01. This file intentionally contains no dataset download
or derived data. A dataset may not enter training until its card, repository license, and
the license of any upstream source have been checked.

| Dataset | Intended use | Dataset card / source | License | Commercial use | Status |
| --- | --- | --- | --- | --- | --- |
| VitaminC | Train/validation/test evidence pairs; retain `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO` when present | https://huggingface.co/datasets/tals/vitaminc ; https://github.com/TalSchuster/VitaminC | UNVERIFIED | UNVERIFIED | Verify the Hugging Face card and upstream repository before download. |
| MNLI | Optional entailment training augmentation | https://huggingface.co/datasets/nyu-mll/multi_nli ; https://cims.nyu.edu/~sbowman/multinli/ | UNVERIFIED | UNVERIFIED | Verify the current dataset card and original corpus terms before download. |
| RAGTruth | Training only; held-out official test for evaluation | https://huggingface.co/datasets/lyab0/RAGTruth ; https://github.com/google-deepmind/long-form-factuality | UNVERIFIED | UNVERIFIED | Verify the dataset card, repository terms, and constituent data sources before download. |

## Excluded datasets

ANLI and DocNLI are excluded from this pipeline and must not be added without a separate
license review.

## Required evidence for final approval

For each accepted dataset, add the checked-on date, exact license text or identifier,
upstream-source obligations, redistribution conditions, attribution requirements, and an
explicit commercial-use determination. Stop work and obtain direction if any source is
non-commercial or the terms cannot be verified.
