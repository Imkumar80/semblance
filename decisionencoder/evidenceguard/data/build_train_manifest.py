"""Compose a balanced, train-only EvidenceGuard manifest from local JSONL inputs."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator


COMPONENTS = {
    "hard_negatives": 30,
    "multichunk": 35,
    "ragtruth_train": 35,
}
COMPONENT_OFFSETS = {name: index for index, name in enumerate(COMPONENTS)}
EVALUATION_SPLITS = {"dev", "val", "validation", "test"}


def _normalized(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, row


def _required_string(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} must be a non-empty string")
    return value


def _prepare_training_row(
    row: dict[str, Any], path: Path, line_number: int, component: str
) -> dict[str, Any] | None:
    where = f"{path}:{line_number}"
    split = row.get("split")
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"{where}: split is required to verify train-only input")
    if split != "train":
        return None

    _required_string(row, "id", where)
    _required_string(row, "claim", where)
    _required_string(row, "source", where)
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError(f"{where}: label must be integer 0 or 1")

    chunks = row.get("chunks")
    if chunks is None:
        premise = _required_string(row, "premise", where)
        chunks = [premise]
    if (
        not isinstance(chunks, list)
        or not chunks
        or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
    ):
        raise ValueError(f"{where}: chunks must be a non-empty list of non-empty strings")
    if component == "multichunk" and len(chunks) < 2:
        raise ValueError(f"{where}: multichunk rows must contain at least two chunks")

    prepared = dict(row)
    prepared["chunks"] = chunks
    prepared["source_component"] = component
    return prepared


def _collect_evaluation_keys(paths: list[Path]) -> tuple[dict[str, set[str]], int]:
    keys = {"ids": set(), "claims": set(), "sources": set()}
    count = 0
    for path in paths:
        for line_number, row in _read_jsonl(path):
            where = f"{path}:{line_number}"
            split = row.get("split")
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"{where}: split is required in evaluation manifests")
            if split == "train":
                continue
            if split not in EVALUATION_SPLITS:
                raise ValueError(f"{where}: unrecognized evaluation split {split!r}")
            keys["ids"].add(_required_string(row, "id", where))
            parent_id = row.get("parent_id")
            if isinstance(parent_id, str) and parent_id.strip():
                keys["ids"].add(parent_id)
            keys["claims"].add(_normalized(_required_string(row, "claim", where)))
            keys["sources"].add(_normalized(_required_string(row, "source", where)))
            count += 1
    if count == 0:
        raise ValueError("evaluation manifests contain no validation/test rows")
    return keys, count


def _validate_no_evaluation_overlap(
    row: dict[str, Any], keys: dict[str, set[str]], component: str, where: str
) -> None:
    identifiers = {row["id"]}
    parent_id = row.get("parent_id")
    if isinstance(parent_id, str) and parent_id.strip():
        identifiers.add(parent_id)
    if identifiers.intersection(keys["ids"]):
        raise ValueError(f"{where}: {component} ID overlaps an evaluation ID")
    if _normalized(row["claim"]) in keys["claims"]:
        raise ValueError(f"{where}: {component} claim overlaps an evaluation claim")
    if _normalized(row["source"]) in keys["sources"]:
        raise ValueError(f"{where}: {component} source overlaps an evaluation source")


def _count_training_rows(
    path: Path, component: str, evaluation_keys: dict[str, set[str]]
) -> int:
    count = 0
    seen_ids = set()
    for line_number, row in _read_jsonl(path):
        prepared = _prepare_training_row(row, path, line_number, component)
        if prepared is None:
            continue
        _validate_no_evaluation_overlap(
            prepared, evaluation_keys, component, f"{path}:{line_number}"
        )
        if prepared["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate training ID {prepared['id']!r}")
        seen_ids.add(prepared["id"])
        count += 1
    return count


def _allocation(total_rows: int) -> dict[str, int]:
    if total_rows <= 0 or total_rows % 20:
        raise ValueError("total_rows must be a positive multiple of 20 for exact 30/35/35 shares")
    return {
        component: total_rows * percentage // 100
        for component, percentage in COMPONENTS.items()
    }


def _sample_training_rows(
    path: Path,
    component: str,
    sample_size: int,
    seed: int,
    expected_count: int,
    evaluation_keys: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rng = random.Random(seed + COMPONENT_OFFSETS[component])
    reservoir: list[dict[str, Any]] = []
    seen_ids = set()
    count = 0
    for line_number, row in _read_jsonl(path):
        prepared = _prepare_training_row(row, path, line_number, component)
        if prepared is None:
            continue
        _validate_no_evaluation_overlap(
            prepared, evaluation_keys, component, f"{path}:{line_number}"
        )
        if prepared["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate training ID {prepared['id']!r}")
        seen_ids.add(prepared["id"])
        count += 1
        if len(reservoir) < sample_size:
            reservoir.append(prepared)
        else:
            replacement_index = rng.randrange(count)
            if replacement_index < sample_size:
                reservoir[replacement_index] = prepared
    if count != expected_count:
        raise ValueError(f"{path}: row count changed while composing the manifest")
    return reservoir


def compose_manifest(
    input_paths: dict[str, Path],
    evaluation_paths: list[Path],
    total_rows: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample an exact 30/35/35 mixture, rejecting any train/evaluation overlap."""
    if set(input_paths) != set(COMPONENTS):
        raise ValueError(f"input_paths must have exactly these keys: {sorted(COMPONENTS)}")
    evaluation_keys, evaluation_count = _collect_evaluation_keys(evaluation_paths)
    available = {
        component: _count_training_rows(input_paths[component], component, evaluation_keys)
        for component in COMPONENTS
    }
    if total_rows is None:
        maximum = min(
            available[component] * 100 // percentage
            for component, percentage in COMPONENTS.items()
        )
        total_rows = maximum // 20 * 20
        if total_rows == 0:
            raise ValueError(
                "not enough train rows to form the smallest exact mixture (6/7/7); "
                f"available: {available}"
            )
    allocation = _allocation(total_rows)
    insufficient = {
        component: {"required": allocation[component], "available": available[component]}
        for component in COMPONENTS
        if available[component] < allocation[component]
    }
    if insufficient:
        raise ValueError(f"insufficient train rows for requested mixture: {insufficient}")

    rows = []
    for component, sample_size in allocation.items():
        sampled = _sample_training_rows(
            input_paths[component],
            component,
            sample_size,
            seed,
            available[component],
            evaluation_keys,
        )
        for row in sampled:
            row["original_id"] = row["id"]
            row["id"] = f"{component}:{row['id']}"
            row["mixture_share"] = COMPONENTS[component] / 100
            rows.append(row)
    random.Random(seed).shuffle(rows)
    report = {
        "records": len(rows),
        "available_train_rows": available,
        "selected_rows": allocation,
        "evaluation_rows_checked": evaluation_count,
        "seed": seed,
    }
    return rows, report


def write_manifest(rows: list[dict[str, Any]], output_path: Path, force: bool = False) -> None:
    mode = "w" if force else "x"
    with output_path.open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-only 30/35/35 manifest from local JSONL inputs."
    )
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--multichunk", type=Path, required=True)
    parser.add_argument("--ragtruth-train", type=Path, required=True)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="local manifests containing validation/test rows for overlap checks",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--total-rows",
        type=int,
        help="exact output size (multiple of 20); default uses the maximum feasible size",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    parser.add_argument(
        "--confirm-ragtruth-rights",
        action="store_true",
        help="confirm the local RAGTruth-derived input is cleared for this intended use",
    )
    args = parser.parse_args()

    if not args.confirm_ragtruth_rights:
        parser.error(
            "RAGTruth source-document rights are unverified; clear the intended use before "
            "passing --confirm-ragtruth-rights"
        )
    input_paths = {
        "hard_negatives": args.hard_negatives,
        "multichunk": args.multichunk,
        "ragtruth_train": args.ragtruth_train,
    }
    all_inputs = [*input_paths.values(), *args.eval_manifest]
    if args.out.resolve() in {path.resolve() for path in all_inputs}:
        parser.error("--out must not point to any input or evaluation manifest")
    if args.out.exists() and not args.force:
        parser.error(f"output already exists (use --force to replace it): {args.out}")

    rows, report = compose_manifest(
        input_paths,
        args.eval_manifest,
        total_rows=args.total_rows,
        seed=args.seed,
    )
    write_manifest(rows, args.out, force=args.force)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
