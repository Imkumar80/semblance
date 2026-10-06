"""Build review-required distractor stress-test variants from disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_COUNTS = (0, 1, 2, 4, 6, 8)
DEFAULT_POSITIONS = ("first", "middle", "last")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows


def _validate_anchor(row: dict[str, Any], where: str) -> None:
    if row.get("split") != "test":
        raise ValueError(f"{where}: anchor rows must have split='test'")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError(f"{where}: id must be non-empty text")
    if not isinstance(row.get("source"), str) or not row["source"].strip():
        raise ValueError(f"{where}: source must be non-empty text")
    if not isinstance(row.get("claim"), str) or not row["claim"].strip():
        raise ValueError(f"{where}: claim must be non-empty text")
    chunks = row.get("chunks")
    if not isinstance(chunks, list) or not chunks or any(
        not isinstance(chunk, str) or not chunk.strip() for chunk in chunks
    ):
        raise ValueError(f"{where}: chunks must be a non-empty list of text")
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError(f"{where}: label must be integer 0 or 1")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Return standard Okapi BM25 scores using only the Python standard library."""
    if not documents:
        return []
    tokenized = [_tokens(document) for document in documents]
    query_terms = set(_tokens(query))
    lengths = [len(document) for document in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter(
        term for document in tokenized for term in set(document)
    )
    k1, b = 1.5, 0.75
    scores = []
    for document, length in zip(tokenized, lengths):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(documents) - df + 0.5) / (df + 0.5)
            )
            norm = frequency + k1 * (
                1.0 - b + b * length / max(average_length, 1.0)
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / norm
        scores.append(score)
    return scores


def select_unique_source_anchors(
    anchors: list[dict[str, Any]], max_anchors: int | None = None
) -> list[dict[str, Any]]:
    """Keep the first eligible anchor from each source, up to the requested count."""
    selected = []
    seen_sources: set[str] = set()
    for anchor in anchors:
        source = anchor.get("source") or anchor.get("parent_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("unique-source anchor selection requires source or parent_id")
        source_key = source.casefold()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        selected.append(anchor)
        if max_anchors is not None and len(selected) >= max_anchors:
            break
    return selected


def _pool_chunks(
    rows: list[dict[str, Any]], excluded_sources: set[str]
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen_chunks: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"distractor row {index}"
        if row.get("split") != "train":
            raise ValueError(f"{where}: distractor rows must have split='train'")
        source = row.get("source")
        chunks = row.get("chunks")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{where}: source must be non-empty text")
        if not isinstance(chunks, list) or any(
            not isinstance(chunk, str) or not chunk.strip() for chunk in chunks
        ):
            raise ValueError(f"{where}: chunks must be a list of non-empty text")
        if source.casefold() in excluded_sources:
            continue
        for chunk in chunks:
            key = chunk.strip().casefold()
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            candidates.append({"source": source.strip(), "text": chunk.strip()})
    return candidates


def make_variants(
    anchors: list[dict[str, Any]],
    distractor_rows: list[dict[str, Any]],
    counts: tuple[int, ...] = DEFAULT_COUNTS,
    positions: tuple[str, ...] = DEFAULT_POSITIONS,
    distractor_selection: str = "high-bm25",
) -> list[dict[str, Any]]:
    """Create paired conditions; labels are provisional until every context is reviewed."""
    if not counts or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("counts must contain non-negative integers")
    if len(set(counts)) != len(counts):
        raise ValueError("counts must not contain duplicates")
    if not positions or any(position not in DEFAULT_POSITIONS for position in positions):
        raise ValueError(f"positions must be selected from {DEFAULT_POSITIONS}")
    if len(set(positions)) != len(positions):
        raise ValueError("positions must not contain duplicates")
    if distractor_selection not in {"high-bm25", "low-bm25"}:
        raise ValueError("distractor_selection must be 'high-bm25' or 'low-bm25'")

    anchor_ids: set[str] = set()
    anchor_sources: set[str] = set()
    for index, row in enumerate(anchors, 1):
        where = f"anchor row {index}"
        _validate_anchor(row, where)
        if row["id"] in anchor_ids:
            raise ValueError(f"{where}: duplicate anchor id {row['id']!r}")
        anchor_ids.add(row["id"])
        anchor_sources.add(row["source"].casefold())
    pool = _pool_chunks(distractor_rows, anchor_sources)
    if not anchors:
        raise ValueError("at least one test anchor is required")

    output: list[dict[str, Any]] = []
    for anchor in anchors:
        candidates = [
            candidate
            for candidate in pool
            if candidate["text"].casefold()
            not in {chunk.strip().casefold() for chunk in anchor["chunks"]}
        ]
        scores = bm25_scores(anchor["claim"], [item["text"] for item in candidates])
        if distractor_selection == "low-bm25":
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (
                    item[1],
                    hashlib.sha256(
                        f"{anchor['id']}\0{item[0]['source']}\0{item[0]['text']}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                ),
            )
        else:
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (
                    -item[1],
                    item[0]["source"].casefold(),
                    item[0]["text"],
                ),
            )
        # At most one chunk per donor source makes distractors source-diverse.
        ranked_unique: list[dict[str, str]] = []
        donor_sources: set[str] = set()
        for candidate, _score in ranked:
            source_key = candidate["source"].casefold()
            if source_key in donor_sources:
                continue
            donor_sources.add(source_key)
            ranked_unique.append(candidate)

        for count in counts:
            if count > len(ranked_unique):
                raise ValueError(
                    f"anchor {anchor['id']!r} needs {count} distinct-source distractors; "
                    f"only {len(ranked_unique)} are available"
                )
            selected = ranked_unique[:count]
            condition_positions = ("none",) if count == 0 else positions
            for position in condition_positions:
                evidence_chunks = [chunk.strip() for chunk in anchor["chunks"]]
                distractor_chunks = [item["text"] for item in selected]
                if position == "first":
                    chunks = evidence_chunks + distractor_chunks
                    evidence_offset = 0
                elif position == "middle":
                    midpoint = count // 2
                    chunks = (
                        distractor_chunks[:midpoint]
                        + evidence_chunks
                        + distractor_chunks[midpoint:]
                    )
                    evidence_offset = midpoint
                elif position == "last":
                    chunks = distractor_chunks + evidence_chunks
                    evidence_offset = count
                else:
                    chunks = evidence_chunks
                    evidence_offset = 0
                variant_id = f"{anchor['id']}__d{count}__{position}"
                output.append(
                    {
                        "id": variant_id,
                        "anchor_id": anchor["id"],
                        "source": anchor["source"],
                        "claim": anchor["claim"],
                        "chunks": chunks,
                        "label": anchor["label"],
                        "split": "test",
                        "distractor_count": count,
                        "evidence_position": position,
                        "decision_evidence_indices": None,
                        "distractor_sources": [
                            item["source"] for item in selected
                        ],
                        "distractor_selection": distractor_selection,
                        "review_status": "needs_review",
                        "review_note": (
                            "Verify every inserted chunk is irrelevant to this claim and "
                            "that the inherited label remains valid. Annotate "
                            "decision_evidence_indices with every chunk required to "
                            "determine the label; use [] only when no chunk is required."
                        ),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--distractors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--counts",
        "--distractor-counts",
        dest="counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
    )
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument(
        "--unique-anchor-source",
        action="store_true",
        help="select at most one anchor for each source document",
    )
    parser.add_argument(
        "--distractor-selection",
        choices=("high-bm25", "low-bm25"),
        default="high-bm25",
        help="rank donor chunks by lexical overlap (high or low)",
    )
    parser.add_argument(
        "--distractor-task-types",
        choices=("Summary", "QA", "Data2txt"),
        nargs="+",
        help="restrict donor rows to these RAGTruth task types",
    )
    parser.add_argument(
        "--positions",
        choices=DEFAULT_POSITIONS,
        nargs="+",
        default=list(DEFAULT_POSITIONS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_anchors is not None and args.max_anchors <= 0:
        parser.error("--max-anchors must be positive")
    if args.out.resolve() in {args.anchors.resolve(), args.distractors.resolve()}:
        parser.error("output must not overwrite either input")
    if args.out.exists() and not args.force:
        parser.error(f"output exists (use --force to replace it): {args.out}")
    try:
        anchors = _read_jsonl(args.anchors)
        if args.unique_anchor_source:
            anchors = select_unique_source_anchors(anchors, args.max_anchors)
        elif args.max_anchors is not None:
            anchors = anchors[: args.max_anchors]
        distractors = _read_jsonl(args.distractors)
        if args.distractor_task_types:
            allowed_task_types = set(args.distractor_task_types)
            distractors = [
                row for row in distractors if row.get("task_type") in allowed_task_types
            ]
            if not distractors:
                raise ValueError(
                    "no distractor rows match --distractor-task-types"
                )
        rows = make_variants(
            anchors,
            distractors,
            tuple(args.counts),
            tuple(args.positions),
            args.distractor_selection,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "anchors": len({row["anchor_id"] for row in rows}),
                "review_required": True,
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
