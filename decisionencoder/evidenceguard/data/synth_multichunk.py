"""Create deterministic, leak-free synthetic multi-chunk evidence examples.

Each scenario is assigned wholesale to one split, so its bridge entity and source documents
cannot appear in both training and evaluation data.  This generator is for controlled
multi-hop evaluation and training augmentation, not a substitute for hand-checked data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCENARIOS = [
    ("Aster Institute", "Mira Chen", "the Lunar Materials Prize", "Helios Foundation"),
    ("Briar Laboratory", "Omar Shah", "the Coastal Science Medal", "Tidewell Trust"),
    ("Cedar Observatory", "Elena Ruiz", "the Aurora Research Award", "Northstar Fund"),
    ("Dune Archive", "Tariq Ali", "the Heritage Translation Prize", "Atlas Council"),
    ("Elm Conservatory", "Nora Singh", "the Green Cities Fellowship", "Verdant League"),
    ("Fjord Museum", "Leo Martin", "the Maritime History Grant", "Harbor Society"),
]
SPLITS = ("train", "val", "test")


def assign_split(group_id: str) -> str:
    """Stable split assignment based on the group identifier, not record order."""
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return SPLITS[int.from_bytes(digest[:8], "big") % len(SPLITS)]


def make_examples(scenarios: Iterable[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    """Emit supported and contradicted two-hop claims for each independent scenario."""
    output: list[dict[str, Any]] = []
    for institution, researcher, award, funder in scenarios:
        group_id = institution.lower().replace(" ", "-")
        split = assign_split(group_id)
        chunks = [
            f"{researcher} is a researcher at {institution}.",
            f"{institution} received {award} from the {funder}.",
        ]
        supported_claim = f"{researcher}'s institution received {award} from the {funder}."
        contradictory_claim = f"{researcher}'s institution received {award} from an unrelated sponsor."
        for fine_label, claim, label in (
            ("SUPPORTS", supported_claim, 1),
            ("REFUTES", contradictory_claim, 0),
        ):
            output.append(
                {
                    "id": f"synthetic-{group_id}-{fine_label.lower()}",
                    "source": group_id,
                    "group_id": group_id,
                    "split": split,
                    "claim": claim,
                    "chunks": chunks,
                    "premise": "\n\n".join(chunks),
                    "label": label,
                    "fine_label": fine_label,
                    "generator": "synth_multichunk_v1",
                }
            )
    return output


def validate_leak_free(items: Iterable[dict[str, Any]]) -> None:
    """Raise if a source group crosses splits or an ID is repeated."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    identifiers: set[str] = set()
    for item in items:
        if item["id"] in identifiers:
            raise ValueError(f"duplicate synthetic ID: {item['id']}")
        identifiers.add(item["id"])
        group_splits[item["group_id"]].add(item["split"])
    leaked = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"cross-split synthetic group leakage: {leaked}")


def report(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(items)
    return {
        "records": len(records),
        "by_split": dict(sorted(Counter(item["split"] for item in records).items())),
        "by_label": dict(sorted(Counter(item["fine_label"] for item in records).items())),
        "unique_groups": len({item["group_id"] for item in records}),
        "chunk_count": dict(sorted(Counter(len(item["chunks"]) for item in records).items())),
        "claim_chars": {
            "min": min((len(item["claim"]) for item in records), default=0),
            "max": max((len(item["claim"]) for item in records), default=0),
        },
        "leak_free": True,
        "warning": "Synthetic examples are not hand-checked multi-hop evaluation data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("synthetic_multichunk.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("synthetic_multichunk_report.json"))
    args = parser.parse_args()

    items = make_examples(DEFAULT_SCENARIOS)
    validate_leak_free(items)
    args.out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    summary = report(items)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
