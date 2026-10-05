"""Filter component JSONL to train rows disjoint from evaluation manifests."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_train_manifest import _collect_evaluation_keys, _normalized, _read_jsonl


def prepare_train_component(
    input_path: Path,
    evaluation_paths: list[Path],
    output_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    if output_path.resolve() in {
        input_path.resolve(),
        *(path.resolve() for path in evaluation_paths),
    }:
        raise ValueError("output must not overwrite an input or evaluation manifest")
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")

    evaluation_keys, evaluation_count = _collect_evaluation_keys(evaluation_paths)
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for line_number, row in _read_jsonl(input_path):
                where = f"{input_path}:{line_number}"
                split = row.get("split")
                if split not in {"train", "dev", "val", "validation", "test"}:
                    raise ValueError(f"{where}: unrecognized split {split!r}")
                if split != "train":
                    counts["non_train_rows_ignored"] += 1
                    continue
                for field in ("id", "claim", "source"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"{where}: {field} must be a non-empty string")

                ids = {row["id"]}
                parent_id = row.get("parent_id")
                if isinstance(parent_id, str) and parent_id.strip():
                    ids.add(parent_id)
                overlaps = []
                if ids.intersection(evaluation_keys["ids"]):
                    overlaps.append("id")
                if _normalized(row["claim"]) in evaluation_keys["claims"]:
                    overlaps.append("claim")
                if _normalized(row["source"]) in evaluation_keys["sources"]:
                    overlaps.append("source")
                if overlaps:
                    counts["train_rows_removed"] += 1
                    for reason in overlaps:
                        counts[f"removed_for_{reason}_overlap"] += 1
                    continue

                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["train_rows_kept"] += 1
            output.flush()
            os.fsync(output.fileno())

        if not counts["train_rows_kept"]:
            raise ValueError("no leakage-free train rows remain after filtering")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "input_rows": sum(
            counts[key]
            for key in (
                "non_train_rows_ignored",
                "train_rows_removed",
                "train_rows_kept",
            )
        ),
        "evaluation_rows_checked": evaluation_count,
        "train_rows_kept": counts["train_rows_kept"],
        "train_rows_removed": counts["train_rows_removed"],
        "removed_for_overlap": {
            reason: counts[f"removed_for_{reason}_overlap"]
            for reason in ("id", "claim", "source")
        },
        "non_train_rows_ignored": counts["non_train_rows_ignored"],
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a train-only component input with evaluation overlaps removed."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="manifests containing validation/test rows",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args()
    try:
        report = prepare_train_component(
            args.input,
            args.eval_manifest,
            args.out,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
