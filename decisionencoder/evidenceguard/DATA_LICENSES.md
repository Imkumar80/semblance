# EvidenceGuard data-license audit

Status: reviewed 2026-10-01. This file intentionally contains no derived training data.
A dataset may not enter commercial training or redistribution until its card, repository
license, and every applicable upstream-source term have been checked.

| Dataset | Intended use | Dataset card / source | License | Commercial use | Status |
| --- | --- | --- | --- | --- | --- |
| VitaminC | Train/validation/test evidence pairs; retain `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO` when present | [HF card](https://huggingface.co/datasets/tals/vitaminc); [upstream data license](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE) | Dataset annotations: CC BY-SA 3.0 / applicable Wikipedia terms; code: MIT | Yes, conditional | Attribute sources and preserve share-alike obligations. Synthetic examples also modify FEVER material; its terms must accompany any redistribution. |
| MNLI | Optional entailment training augmentation | [HF card](https://huggingface.co/datasets/nyu-mll/multi_nli); [OANC terms](https://anc.org/data/oanc/) | Composite: mostly OANC; fiction includes CC BY-SA 3.0, CC BY 3.0, and US public-domain works | Yes, conditional | OANC explicitly permits commercial use. Retain attribution and share-alike obligations for the identified fiction sources; do not represent the corpus as uniformly licensed. |
| RAGTruth | Training only; held-out official test for evaluation | [upstream repository](https://github.com/ParticleMedia/RAGTruth); [MIT license](https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE) | MIT for the repository and corpus release | UNVERIFIED for embedded source documents | The repository license permits commercial use, but responses are grounded in third-party source documents (including news). Do not use for commercial training or redistribute its raw text until source-level rights are cleared. |

## Excluded datasets

ANLI and DocNLI are excluded from this pipeline and must not be added without a separate
license review.

## Required evidence for final approval

For each accepted dataset, retain the checked-on date, exact license text or identifier,
upstream-source obligations, redistribution conditions, attribution requirements, and an
explicit commercial-use determination. Stop work and obtain direction if any source is
non-commercial. Treat an `UNVERIFIED` entry as blocked for commercial training and
redistribution, rather than as permission.
