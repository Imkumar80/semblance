"""Create train-only two-chunk examples from independent supported VitaminC facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_records(path: Path) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    evaluation_sources: set[str] = set()
    supported_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_facts: set[tuple[str, str, str]] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            split = row.get("split")
            source = row.get("source")
            if split not in {"train", "val", "validation", "dev", "test"}:
                raise ValueError(f"{path}:{line_number}: unsupported split {split!r}")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{path}:{line_number}: source must be non-empty text")
            source = source.strip()
            if split != "train":
                evaluation_sources.add(source.casefold())
                continue
            if row.get("label") != 1:
                continue
            claim, premise = row.get("claim"), row.get("premise")
            if (
                not isinstance(claim, str)
                or not claim.strip()
                or not isinstance(premise, str)
                or not premise.strip()
            ):
                raise ValueError(
                    f"{path}:{line_number}: supported train rows need claim and premise text"
                )
            fact_key = (source.casefold(), claim.casefold(), premise.casefold())
            if fact_key in seen_source_facts:
                continue
            seen_source_facts.add(fact_key)
            supported_by_source[source].append(
                {
                    "id": row.get("id"),
                    "claim": claim.strip(),
                    "premise": premise.strip(),
                }
            )
    return evaluation_sources, supported_by_source


def _stable_id(source: str, index: int, kind: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"vitaminc-mc-train-{digest}-{index:05d}-{kind}"


def _combine_claims(first: str, second: str) -> str:
    return f"{first.rstrip()} Additionally, {second[0].lower() + second[1:]}"


def make_examples(
    input_path: Path,
    max_rows: int = 15_000,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build paired two-chunk hop positives and missing-evidence hard negatives.

    For a page with three distinct supported records, the positive uses two records'
    evidence and joined claims. The negative keeps the first evidence and swaps the
    second evidence for a third page passage, while retaining the two-fact claim.
    Negative examples are marked as synthetic missing-evidence cases for review.
    """
    if max_rows <= 0 or max_rows % 2:
        raise ValueError("max_rows must be a positive even number")
    evaluation_sources, supported_by_source = _read_records(input_path)
    candidates = [
        (source, facts)
        for source, facts in supported_by_source.items()
        if source.casefold() not in evaluation_sources and len(facts) >= 3
    ]
    target_pairs = max_rows // 2
    if len(candidates) < target_pairs:
        raise ValueError(
            f"requested {target_pairs} pairs but only {len(candidates)} train pages "
            "have at least three distinct supported claims"
        )

    rng = random.Random(seed)
    candidates.sort(key=lambda item: item[0].casefold())
    selected = rng.sample(candidates, target_pairs)
    examples = []
    for index, (source, facts) in enumerate(selected):
        first, second, distractor = rng.sample(facts, 3)
        combined_claim = _combine_claims(first["claim"], second["claim"])
        shared = {
            "source": source,
            "split": "train",
            "parent_id": f"vitaminc-page-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}",
            "claim": combined_claim,
            "generator": "vitaminc_two_fact_evidence_drop_v1",
        }
        examples.append(
            {
                **shared,
                "id": _stable_id(source, index, "hop_pos"),
                "chunks": [first["premise"], second["premise"]],
                "label": 1,
                "type": "hop_pos",
                "source_fact_ids": [first["id"], second["id"]],
            }
        )
        examples.append(
            {
                **shared,
                "id": _stable_id(source, index, "hop_drop"),
                "chunks": [first["premise"], distractor["premise"]],
                "label": 0,
                "type": "hop_drop",
                "source_fact_ids": [first["id"], distractor["id"]],
                "dropped_fact_id": second["id"],
                "review_note": (
                    "Synthetic missing-evidence contrast. Verify distractor does not "
                    "support the second clause before use in a benchmark."
                ),
            }
        )
    rng.shuffle(examples)
    report = {
        "records": len(examples),
        "by_type": {
            "hop_pos": target_pairs,
            "hop_drop": target_pairs,
        },
        "candidate_train_pages": len(candidates),
        "selected_train_pages": target_pairs,
        "evaluation_sources_excluded": len(evaluation_sources),
        "seed": seed,
        "warning": (
            "Synthetic positives and missing-evidence negatives require review; "
            "VitaminC support labels do not prove the composed claim or distractor semantics."
        ),
    }
    return examples, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in (args.out, args.report):
        if path.resolve() == args.input.resolve():
            parser.error("outputs must not overwrite the VitaminC input")
        if path.exists() and not args.force:
            parser.error(f"output exists (use --force to replace it): {path}")
    if args.out.resolve() == args.report.resolve():
        parser.error("--out and --report must be different paths")
    try:
        examples, report = make_examples(args.input, args.max_rows, args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
