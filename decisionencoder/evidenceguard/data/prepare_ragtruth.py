"""Normalize local RAGTruth JSONL files into EvidenceGuard train/test JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


PASSAGE_BOUNDARY_RE = re.compile(r"(?i)(?=passage\s+\d+\s*:)")
ALLOWED_SPLITS = {"train", "test"}
ALLOWED_QUALITY = {"good", "incorrect_refusal", "truncated"}


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


def _required_text(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} must be a non-empty string")
    return value.strip()


def _source_chunks(source_info: Any, where: str) -> list[str]:
    if isinstance(source_info, str):
        chunks = [source_info.strip()]
    elif isinstance(source_info, dict):
        passages = source_info.get("passages")
        if isinstance(passages, str):
            split_passages = [
                passage.strip()
                for passage in PASSAGE_BOUNDARY_RE.split(passages)
                if passage.strip()
            ]
            chunks = split_passages or [passages.strip()]
        else:
            chunks = []
            for key, value in source_info.items():
                if isinstance(value, str) and value.strip():
                    chunks.append(f"{key}: {value.strip()}")
                elif isinstance(value, list):
                    chunks.extend(
                        f"{key}: {item.strip()}"
                        for item in value
                        if isinstance(item, str) and item.strip()
                    )
            if not chunks:
                chunks = [json.dumps(source_info, ensure_ascii=False, sort_keys=True)]
    else:
        raise ValueError(f"{where}: source_info must be a string or object")
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise ValueError(f"{where}: source_info contains no usable evidence text")
    return chunks


def _load_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources = {}
    for line_number, row in _read_jsonl(path):
        where = f"{path}:{line_number}"
        source_id = _required_text(row.get("source_id"), "source_id", where)
        if source_id in sources:
            raise ValueError(f"{where}: duplicate source_id {source_id!r}")
        source_origin = _required_text(row.get("source"), "source", where)
        chunks = _source_chunks(row.get("source_info"), where)
        sources[source_id] = {
            "source_origin": source_origin,
            "task_type": row.get("task_type"),
            "chunks": chunks,
        }
    if not sources:
        raise ValueError(f"{path}: no source records found")
    return sources


def _response_source_splits(path: Path) -> dict[str, set[str]]:
    splits_by_source: dict[str, set[str]] = {}
    response_ids = set()
    for line_number, row in _read_jsonl(path):
        where = f"{path}:{line_number}"
        response_id = _required_text(row.get("id"), "id", where)
        if response_id in response_ids:
            raise ValueError(f"{where}: duplicate response id {response_id!r}")
        response_ids.add(response_id)
        source_id = _required_text(row.get("source_id"), "source_id", where)
        split = row.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{where}: unsupported official split {split!r}")
        splits_by_source.setdefault(source_id, set()).add(split)
    if not response_ids:
        raise ValueError(f"{path}: no response records found")
    return splits_by_source


def _normalize_response(
    row: dict[str, Any],
    source: dict[str, Any],
    line_number: int,
    response_path: Path,
) -> dict[str, Any] | None:
    where = f"{response_path}:{line_number}"
    response_id = _required_text(row.get("id"), "id", where)
    source_id = _required_text(row.get("source_id"), "source_id", where)
    response = _required_text(row.get("response"), "response", where)
    labels = row.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"{where}: labels must be a list")
    quality = row.get("quality")
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"{where}: unrecognized quality value {quality!r}")
    if quality != "good":
        return None
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError(f"{where}: each hallucination label must be an object")
        start, end, text = label.get("start"), label.get("end"), label.get("text")
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(text, str)
            or start < 0
            or end < start
            or end > len(response)
        ):
            raise ValueError(f"{where}: malformed hallucination span {label!r}")

    is_supported = not labels
    source_key = f"RAGTruth:{source_id}"
    return {
        "id": f"ragtruth-response-{response_id}",
        "parent_id": f"ragtruth-source-{source_id}",
        "source": source_key,
        "source_origin": source["source_origin"],
        "source_id": source_id,
        "task_type": source["task_type"],
        "model": row.get("model"),
        "claim": response,
        "chunks": source["chunks"],
        "label": int(is_supported),
        "fine_label": "SUPPORTED" if is_supported else "UNSUPPORTED",
        "hallucination_spans": labels,
        "split": row["split"],
        "quality": quality,
        "dataset": "RAGTruth",
        "license_use": "user_confirmed_local_training",
    }


def prepare_ragtruth(
    responses_path: Path,
    sources_path: Path,
    train_output: Path,
    test_output: Path,
    confirm_local_training_rights: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write isolated official train/test outputs, excluding cross-split source groups."""
    if not confirm_local_training_rights:
        raise ValueError(
            "RAGTruth's embedded source-document rights require explicit local-training "
            "confirmation; repository MIT licensing alone is insufficient"
        )
    resolved_outputs = {train_output.resolve(), test_output.resolve()}
    if len(resolved_outputs) != 2:
        raise ValueError("train and test outputs must be different paths")
    if resolved_outputs.intersection(
        {responses_path.resolve(), sources_path.resolve()}
    ):
        raise ValueError("outputs must not overwrite raw RAGTruth input files")
    existing = [path for path in (train_output, test_output) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"output already exists: {existing[0]}")

    sources = _load_sources(sources_path)
    splits_by_source = _response_source_splits(responses_path)
    cross_split_sources = {
        source_id
        for source_id, splits in splits_by_source.items()
        if len(splits) > 1
    }
    missing_sources = set(splits_by_source).difference(sources)
    if missing_sources:
        raise ValueError(
            f"response records reference missing source IDs (first 10): "
            f"{sorted(missing_sources)[:10]}"
        )

    counts: Counter[str] = Counter()
    temporary_paths: list[Path] = []
    final_paths = {"train": train_output, "test": test_output}
    streams = {}
    try:
        for split, output_path in final_paths.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            )
            streams[split] = handle
            temporary_paths.append(Path(handle.name))

        for line_number, row in _read_jsonl(responses_path):
            split = row["split"]
            source_id = str(row["source_id"])
            if split == "train" and source_id in cross_split_sources:
                counts["train_rows_removed_cross_split_source"] += 1
                continue
            if row.get("quality") != "good":
                if row.get("quality") not in ALLOWED_QUALITY:
                    raise ValueError(
                        f"{responses_path}:{line_number}: unrecognized quality value "
                        f"{row.get('quality')!r}"
                    )
                counts[f"{split}_rows_removed_quality"] += 1
                continue
            normalized = _normalize_response(
                row, sources[source_id], line_number, responses_path
            )
            if normalized is None:
                continue
            streams[split].write(json.dumps(normalized, ensure_ascii=False) + "\n")
            counts[f"{split}_rows"] += 1
            counts[f"{split}_supported"] += normalized["label"]
            counts[f"{split}_unsupported"] += 1 - normalized["label"]

        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        streams.clear()

        if not counts["train_rows"] or not counts["test_rows"]:
            raise ValueError(
                "both train and held-out test outputs must contain at least one good row"
            )
        for split, output_path in final_paths.items():
            os.replace(temporary_paths.pop(0), output_path)
        return {
            "output_rows": {
                "train": counts["train_rows"],
                "test": counts["test_rows"],
            },
            "labels": {
                "train": {
                    "supported": counts["train_supported"],
                    "unsupported": counts["train_unsupported"],
                },
                "test": {
                    "supported": counts["test_supported"],
                    "unsupported": counts["test_unsupported"],
                },
            },
            "quality_rows_removed": {
                "train": counts["train_rows_removed_quality"],
                "test": counts["test_rows_removed_quality"],
            },
            "train_rows_removed_cross_split_source": counts[
                "train_rows_removed_cross_split_source"
            ],
            "cross_split_source_groups": len(cross_split_sources),
            "train_output": str(train_output),
            "test_output": str(test_output),
        }
    finally:
        for stream in streams.values():
            stream.close()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize official RAGTruth train/test JSONL into EvidenceGuard rows."
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument(
        "--confirm-local-training-rights",
        action="store_true",
        help="confirm source-level terms were cleared for this local training use",
    )
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args()
    try:
        report = prepare_ragtruth(
            args.responses,
            args.sources,
            args.out_train,
            args.out_test,
            confirm_local_training_rights=args.confirm_local_training_rights,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
