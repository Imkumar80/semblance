"""Evaluate joint, chunk-max, and BM25-prefiltered EvidenceGuard scoring."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from decisionencoder.evidenceguard.data.build_distractor_eval import (
    _read_jsonl,
    bm25_scores,
)
from decisionencoder.modeling import ModernBertForGroundedness


def load_reviewed_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: no evaluation rows found")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"{path}:{index}"
        if row.get("split") != "test":
            raise ValueError(f"{where}: evaluator accepts split='test' rows only")
        if row.get("review_status") != "approved":
            raise ValueError(f"{where}: review_status must be 'approved'")
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"{where}: id must be non-empty text")
        if row["id"] in seen_ids:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
        if not isinstance(row.get("claim"), str) or not row["claim"].strip():
            raise ValueError(f"{where}: claim must be non-empty text")
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks or any(
            not isinstance(chunk, str) or not chunk.strip() for chunk in chunks
        ):
            raise ValueError(f"{where}: chunks must be a non-empty list of text")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{where}: label must be integer 0 or 1")
        if type(row.get("distractor_count")) is not int or row["distractor_count"] < 0:
            raise ValueError(f"{where}: distractor_count must be non-negative integer")
        if row.get("evidence_position") not in {"none", "first", "middle", "last"}:
            raise ValueError(f"{where}: invalid evidence_position")
        if not isinstance(row.get("anchor_id"), str) or not row["anchor_id"]:
            raise ValueError(f"{where}: anchor_id must be non-empty text")
        evidence_indices = row.get("decision_evidence_indices")
        if not isinstance(evidence_indices, list) or any(
            type(index) is not int or index < 0 or index >= len(chunks)
            for index in evidence_indices
        ):
            raise ValueError(
                f"{where}: decision_evidence_indices must list valid chunk indexes"
            )
        if len(set(evidence_indices)) != len(evidence_indices):
            raise ValueError(f"{where}: decision_evidence_indices has duplicates")
    return rows


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same non-zero length")
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "n": len(labels),
        "accuracy": (true_positive + true_negative) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def paired_bootstrap_delta(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the per-anchor accuracy difference (first minus second)."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    by_group: dict[str, dict[str, dict[str, tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        anchor_id = row["anchor_id"]
        method = row["method"]
        if method in {first, second}:
            group = row.get("bootstrap_group", row.get("source_group", anchor_id))
            by_group[group][anchor_id][method] = (row["label"], row["prediction"])
    group_deltas = []
    for anchors in by_group.values():
        deltas = [
            int(values[first][0] == values[first][1])
            - int(values[second][0] == values[second][1])
            for values in anchors.values()
            if first in values and second in values
        ]
        if deltas:
            group_deltas.append(sum(deltas) / len(deltas))
    if not group_deltas:
        raise ValueError("no paired anchor predictions found")
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choices(group_deltas, k=len(group_deltas))) / len(group_deltas)
        for _ in range(iterations)
    )
    low_index = max(0, int(0.025 * iterations))
    high_index = min(iterations - 1, int(0.975 * iterations))
    return {
        "delta_accuracy": sum(group_deltas) / len(group_deltas),
        "ci95_low": samples[low_index],
        "ci95_high": samples[high_index],
        "paired_groups": len(group_deltas),
    }


class EvidenceGuardScorer:
    def __init__(
        self,
        model_path: str,
        device_name: str = "auto",
        max_length: int = 2048,
        prefilter_k: int = 4,
    ) -> None:
        if max_length < 2 or prefilter_k <= 0:
            raise ValueError("max_length must be at least 2 and prefilter_k positive")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but is not available")
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ModernBertForGroundedness.from_pretrained(
            model_path,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.prefilter_k = prefilter_k
        max_positions = getattr(self.model.config, "max_position_embeddings", max_length)
        if max_length > max_positions:
            raise ValueError(
                f"max_length={max_length} exceeds model max_position_embeddings="
                f"{max_positions}"
            )

    @torch.inference_mode()
    def score(self, claim: str, chunks: list[str]) -> float:
        context = "\n\n".join(chunks)
        encoded = self.tokenizer(
            f"Context: {context}",
            f"Claim: {claim}",
            return_tensors="pt",
        )
        sequence_length = encoded["input_ids"].shape[1]
        if sequence_length > self.max_length:
            raise ValueError(
                f"encoded pair has {sequence_length} tokens, exceeding "
                f"--max-length={self.max_length}; increase the limit or shorten "
                "the evaluation context rather than silently truncating evidence"
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits
        return float(torch.sigmoid(logits).item())


def score_rows(
    rows: list[dict[str, Any]],
    scorer: EvidenceGuardScorer,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        claim, chunks = row["claim"], row["chunks"]
        bm25_ranked = sorted(
            zip(range(len(chunks)), chunks, bm25_scores(claim, chunks)),
            key=lambda item: (-item[2], item[0]),
        )
        selected_pairs = bm25_ranked[: scorer.prefilter_k]
        selected_indices = sorted(index for index, _chunk, _score in selected_pairs)
        selected = [chunks[index] for index in selected_indices]
        if not selected:
            raise ValueError(f"{row['id']}: BM25 prefilter selected no evidence chunks")
        gold_evidence = set(row["decision_evidence_indices"])
        retained_evidence = gold_evidence.intersection(selected_indices)
        evidence_recall = (
            len(retained_evidence) / len(gold_evidence) if gold_evidence else None
        )
        complete_evidence_retained = (
            gold_evidence.issubset(selected_indices) if gold_evidence else None
        )
        probabilities = {
            "joint": scorer.score(claim, chunks),
            "chunk_max": max(scorer.score(claim, [chunk]) for chunk in chunks),
            "bm25_joint": scorer.score(claim, selected),
        }
        for method, probability in probabilities.items():
            results.append(
                {
                    "id": row["id"],
                    "anchor_id": row["anchor_id"],
                    "bootstrap_group": row.get("source", row["anchor_id"]),
                    "method": method,
                    "distractor_count": row["distractor_count"],
                    "evidence_position": row["evidence_position"],
                    "label": row["label"],
                    "probability": probability,
                    "prediction": int(probability >= threshold),
                    "bm25_decision_evidence_recall": (
                        evidence_recall if method == "bm25_joint" else None
                    ),
                    "bm25_complete_decision_evidence_retained": (
                        complete_evidence_retained if method == "bm25_joint" else None
                    ),
                }
            )
    return results


def summarize(
    predictions: list[dict[str, Any]], bootstrap_iterations: int, seed: int
) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    all_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in predictions:
        key = (
            result["method"],
            result["distractor_count"],
            result["evidence_position"],
        )
        groups[key].append(result)
        all_by_method[result["method"]].append(result)
    by_condition = []
    for (method, count, position), rows in sorted(groups.items()):
        item = {
            "method": method,
            "distractor_count": count,
            "evidence_position": position,
            **binary_metrics(
                [row["label"] for row in rows],
                [row["prediction"] for row in rows],
            ),
        }
        if method == "bm25_joint":
            evidence_rows = [
                row
                for row in rows
                if row["bm25_decision_evidence_recall"] is not None
            ]
            item["decision_evidence_recall"] = {
                "annotated_examples": len(rows),
                "scorable_examples": len(evidence_rows),
                "mean_recall": (
                    sum(row["bm25_decision_evidence_recall"] for row in evidence_rows)
                    / len(evidence_rows)
                    if evidence_rows
                    else None
                ),
                "complete_retention_rate": (
                    sum(
                        row["bm25_complete_decision_evidence_retained"]
                        for row in evidence_rows
                    )
                    / len(evidence_rows)
                    if evidence_rows
                    else None
                ),
            }
        if method == "joint":
            corresponding = [
                row
                for row in predictions
                if row["distractor_count"] == count
                and row["evidence_position"] == position
                and row["method"] in {"joint", "chunk_max"}
            ]
            item["paired_delta_vs_chunk_max"] = paired_bootstrap_delta(
                corresponding,
                "joint",
                "chunk_max",
                iterations=bootstrap_iterations,
                seed=seed,
            )
        by_condition.append(item)
    return {
        "overall": {
            method: binary_metrics(
                [row["label"] for row in rows],
                [row["prediction"] for row in rows],
            )
            for method, rows in sorted(all_by_method.items())
        },
        "by_condition": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prefilter-k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.manifest.resolve():
        parser.error("--out must not overwrite --manifest")
    if args.out.exists() and not args.force:
        parser.error(f"output exists (use --force to replace it): {args.out}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    try:
        rows = load_reviewed_rows(args.manifest)
        scorer = EvidenceGuardScorer(
            args.model,
            device_name=args.device,
            max_length=args.max_length,
            prefilter_k=args.prefilter_k,
        )
        predictions = score_rows(rows, scorer, threshold=args.threshold)
        report = {
            "model": args.model,
            "threshold": args.threshold,
            "max_length": args.max_length,
            "prefilter_k": args.prefilter_k,
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "n_examples": len(rows),
            **summarize(predictions, args.bootstrap_iterations, args.seed),
            "predictions": predictions,
        }
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_examples", "overall")}, indent=2))


if __name__ == "__main__":
    main()
