"""Create balanced, source-disjoint EvidenceGuard review candidate sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from decisionencoder.evidenceguard.data.build_distractor_eval import _read_jsonl


def _rank(seed: int, *parts: str) -> str:
    value = "\0".join((str(seed), *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_source_rows(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"anchor row {index}"
        if row.get("split") != "test":
            raise ValueError(f"{where}: input rows must have split='test'")
        for field in ("id", "source", "claim"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{where}: {field} must be non-empty text")
        if row["id"] in seen_ids:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks or any(
            not isinstance(chunk, str) or not chunk.strip() for chunk in chunks
        ):
            raise ValueError(f"{where}: chunks must be a non-empty list of text")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{where}: label must be integer 0 or 1")


def _select_label_rows(
    by_source: dict[str, dict[int, list[dict[str, Any]]]],
    label: int,
    count: int,
    seed: int,
    excluded_sources: set[str],
) -> list[dict[str, Any]]:
    eligible = [
        source
        for source, label_rows in by_source.items()
        if source not in excluded_sources and label_rows.get(label)
    ]
    eligible.sort(key=lambda source: _rank(seed, "source", str(label), source))
    if len(eligible) < count:
        raise ValueError(
            f"need {count} distinct sources for label {label}, found {len(eligible)}"
        )
    selected = []
    for source in eligible[:count]:
        source_rows = by_source[source][label]
        source_rows.sort(key=lambda row: _rank(seed, "row", source, row["id"]))
        selected.append(source_rows[0])
    return selected


def _candidate(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "anchor_id": row["id"],
        "source": row["source"],
        "source_id": row.get("source_id"),
        "dataset": row.get("dataset"),
        "task_type": row.get("task_type"),
        "claim": row["claim"],
        "chunks": row["chunks"],
        "label": row["label"],
        "fine_label": row.get("fine_label"),
        "split": split,
        "distractor_count": 0,
        "evidence_position": "none",
        "decision_evidence_indices": None,
        "distractor_sources": [],
        "review_status": "needs_review",
        "review_note": (
            "Verify the claim label and every context chunk. Annotate "
            "decision_evidence_indices with all chunks needed to determine the label; "
            "use [] only when no chunk is needed. Mark approved only after review."
        ),
    }


def build_candidate_splits(
    anchors: list[dict[str, Any]],
    pilot_rows: list[dict[str, Any]],
    anchors_per_split: int = 100,
    unsupported_per_split: int = 50,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if anchors_per_split <= 0 or unsupported_per_split <= 0:
        raise ValueError("split sizes and unsupported count must be positive")
    if unsupported_per_split >= anchors_per_split:
        raise ValueError("each split must include at least one supported anchor")
    _validate_source_rows(anchors)

    excluded_sources: set[str] = set()
    excluded_ids: set[str] = set()
    for row in pilot_rows:
        source = row.get("source")
        anchor_id = row.get("anchor_id")
        if isinstance(source, str):
            excluded_sources.add(source.casefold())
        if isinstance(anchor_id, str):
            excluded_ids.add(anchor_id)

    by_source: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in anchors:
        if row["id"] not in excluded_ids:
            by_source[row["source"].casefold()][row["label"]].append(row)

    negative_count = 2 * unsupported_per_split
    positive_per_split = anchors_per_split - unsupported_per_split
    positive_count = 2 * positive_per_split
    negatives = _select_label_rows(
        by_source, 0, negative_count, seed, excluded_sources
    )
    negative_sources = {row["source"].casefold() for row in negatives}
    positives = _select_label_rows(
        by_source,
        1,
        positive_count,
        seed,
        excluded_sources | negative_sources,
    )

    def split_class(rows: list[dict[str, Any]], per_split: int) -> tuple[list, list]:
        ordered = sorted(
            rows,
            key=lambda row: _rank(seed, "partition", str(row["label"]), row["source"].casefold()),
        )
        return ordered[:per_split], ordered[per_split:]

    dev_neg, test_neg = split_class(negatives, unsupported_per_split)
    dev_pos, test_pos = split_class(positives, positive_per_split)

    development = [
        _candidate(row, "development") for row in (*dev_neg, *dev_pos)
    ]
    test = [_candidate(row, "test") for row in (*test_neg, *test_pos)]
    development.sort(key=lambda row: _rank(seed, "output-dev", row["id"]))
    test.sort(key=lambda row: _rank(seed, "output-test", row["id"]))
    return development, test


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-split", type=int, default=100)
    parser.add_argument("--unsupported-per-split", type=int, default=50)
    parser.add_argument(
        "--split",
        choices=("both", "development", "test"),
        default="both",
        help="write one split only to avoid changing the other split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_paths = {args.anchors.resolve(), args.pilot_manifest.resolve()}
    if args.out_dir.resolve() in input_paths:
        parser.error("--out-dir must not be either input file")
    development_path = args.out_dir / "development_needs_review.jsonl"
    test_path = args.out_dir / "locked_test_needs_review.jsonl"
    output_paths = {
        "development": development_path,
        "test": test_path,
    }
    selected_splits = (
        ("development", "test") if args.split == "both" else (args.split,)
    )
    if not args.force and any(
        output_paths[split].exists() for split in selected_splits
    ):
        parser.error("candidate output exists (use --force to replace it)")
    try:
        development, test = build_candidate_splits(
            _read_jsonl(args.anchors),
            _read_jsonl(args.pilot_manifest),
            anchors_per_split=args.anchors_per_split,
            unsupported_per_split=args.unsupported_per_split,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"development": development, "test": test}
    for split in selected_splits:
        _write_jsonl(output_paths[split], split_rows[split])
    report = {
        split: {
            "path": str(output_paths[split]),
            "rows": len(split_rows[split]),
        }
        for split in selected_splits
    }
    report["status"] = "needs_review"
    print(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
