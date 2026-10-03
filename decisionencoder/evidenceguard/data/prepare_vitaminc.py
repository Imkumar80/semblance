"""Normalize VitaminC records to the EvidenceGuard JSONL schema.

Binary label: 1 = SUPPORTS, 0 = REFUTES or NOT_ENOUGH_INFO.  ``fine_label`` preserves
the source label.  ``--local`` makes the adapter testable without downloading a dataset.
Run ``--show-columns`` before a full Hugging Face conversion to verify the live schema.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HF_SPLITS = {"train": "train", "validation": "val", "test": "test"}
REQUIRED_COLUMNS = {"claim", "evidence", "label", "page"}
SUPPORTED_LABEL = "SUPPORTS"


def _require_columns(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(rows)
    if not materialized:
        return materialized
    missing = REQUIRED_COLUMNS.difference(materialized[0])
    if missing:
        raise ValueError(f"VitaminC rows are missing required columns: {sorted(missing)}")
    return materialized


def convert(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Convert one verified VitaminC split without silently dropping source labels."""
    records = _require_columns(rows)
    refuting: dict[tuple[str, str], str] = {}
    for row in records:
        if row["label"] == "REFUTES":
            refuting.setdefault((row["page"], row["claim"]), row["evidence"])

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in records:
        key = (row["page"], row["claim"], row["evidence"])
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, Any] = {
            "id": f"{split}-{row.get('case_id', len(output))}-{len(output)}",
            "source": row["page"],
            "premise": row["evidence"],
            "claim": row["claim"],
            "label": int(row["label"] == SUPPORTED_LABEL),
            "fine_label": row["label"],
            "split": split,
        }
        refuting_premise = refuting.get((row["page"], row["claim"]))
        if row["label"] == SUPPORTED_LABEL and refuting_premise and refuting_premise != row["evidence"]:
            item["refuting_premise"] = refuting_premise
        output.append(item)
    return output


def drop_cross_split_pages(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove train/validation pages that overlap a later official split."""
    records = list(items)
    splits_by_page: dict[str, set[str]] = defaultdict(set)
    for item in records:
        splits_by_page[item["source"]].add(item["split"])
    return [
        item
        for item in records
        if not (
            (item["split"] == "train" and {"val", "test"}.intersection(splits_by_page[item["source"]]))
            or (item["split"] == "val" and "test" in splits_by_page[item["source"]])
        )
    ]


def _write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("vitaminc_norm.jsonl"))
    parser.add_argument("--local", type=Path, help="VitaminC-style JSONL fixture; rows become train")
    parser.add_argument("--show-columns", action="store_true")
    args = parser.parse_args()

    if args.local:
        rows = [json.loads(line) for line in args.local.read_text(encoding="utf-8").splitlines() if line]
        if args.show_columns:
            print("local", sorted(rows[0]) if rows else [])
            return
        items = convert(rows, "train")
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise SystemExit("Install the `datasets` package before loading VitaminC, or use --local.") from error
        items = []
        for hf_split, normalized_split in HF_SPLITS.items():
            # Schema inspection must not materialize a 100K+ row split locally.
            dataset = load_dataset("tals/vitaminc", split=hf_split, streaming=args.show_columns)
            if args.show_columns:
                first_row = next(iter(dataset))
                print(hf_split, sorted(first_row), first_row)
                return
            items.extend(convert(dataset, normalized_split))

    before = len(items)
    items = drop_cross_split_pages(items)
    _write_jsonl(args.out, items)
    print(f"rows {before} -> {len(items)} after cross-split page removal")
    print(f"label=1: {sum(item['label'] for item in items)}  label=0: {sum(1 - item['label'] for item in items)}")
    print(f"with refuting_premise: {sum('refuting_premise' in item for item in items)}")


if __name__ == "__main__":
    main()
