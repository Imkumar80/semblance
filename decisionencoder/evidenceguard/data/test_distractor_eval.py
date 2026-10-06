import unittest
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from decisionencoder.evidenceguard.data.build_distractor_eval import (
    bm25_scores,
    main as build_distractor_eval_main,
    make_variants,
    select_unique_source_anchors,
)
from decisionencoder.evidenceguard.evaluate_distractor import (
    binary_metrics,
    load_reviewed_rows,
    paired_bootstrap_delta,
    score_rows,
)


def _anchor(**overrides):
    row = {
        "id": "test-anchor",
        "source": "anchor-page",
        "claim": "The Orion probe reached Mars in 2030.",
        "chunks": ["The Orion probe reached Mars.", "It arrived in 2030."],
        "label": 1,
        "split": "test",
    }
    row.update(overrides)
    return row


def _distractors(count=4):
    return [
        {
            "id": f"train-{index}",
            "source": f"donor-page-{index}",
            "chunks": [f"Unrelated archival material about topic {index}."],
            "split": "train",
        }
        for index in range(count)
    ]


class DistractorEvalTests(unittest.TestCase):
    def test_cli_limits_unique_source_anchors_and_accepts_counts_alias(self):
        anchors = [
            _anchor(id="same-source-first", source="source-0"),
            _anchor(id="same-source-second", source="source-0"),
            _anchor(id="other-source", source="source-1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors_path = root / "anchors.jsonl"
            distractors_path = root / "distractors.jsonl"
            output_path = root / "output.jsonl"
            anchors_path.write_text(
                "".join(json.dumps(row) + "\n" for row in anchors),
                encoding="utf-8",
            )
            distractors_path.write_text(
                "".join(json.dumps(row) + "\n" for row in _distractors()),
                encoding="utf-8",
            )
            arguments = [
                "build_distractor_eval",
                "--anchors",
                str(anchors_path),
                "--distractors",
                str(distractors_path),
                "--out",
                str(output_path),
                "--max-anchors",
                "2",
                "--unique-anchor-source",
                "--distractor-counts",
                "0",
                "2",
            ]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                build_distractor_eval_main()

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["anchor_id"] for row in rows},
            {"same-source-first", "other-source"},
        )
        self.assertEqual(len({row["source"] for row in rows if row["distractor_count"] == 0}), 2)

    def test_low_bm25_selection_prefers_lexically_unrelated_donors(self):
        donors = [
            {
                "id": "relevant",
                "source": "relevant-page",
                "chunks": ["The Orion probe reached Mars in 2030."],
                "split": "train",
            },
            *_distractors(1),
        ]
        rows = make_variants(
            [_anchor()],
            donors,
            counts=(1,),
            positions=("first",),
            distractor_selection="low-bm25",
        )
        self.assertEqual(rows[0]["distractor_sources"], ["donor-page-0"])
        self.assertEqual(rows[0]["distractor_selection"], "low-bm25")

    def test_unique_source_anchor_selection_keeps_first_source_occurrence(self):
        anchors = [
            _anchor(id="first", source="same-source"),
            _anchor(id="second", source="same-source"),
            _anchor(id="third", source="different-source"),
        ]
        selected = select_unique_source_anchors(anchors, max_anchors=2)
        self.assertEqual([row["id"] for row in selected], ["first", "third"])

    def test_creates_nested_source_diverse_conditions_and_positions(self):
        rows = make_variants(
            [_anchor()],
            _distractors(),
            counts=(0, 2),
            positions=("first", "middle", "last"),
        )
        self.assertEqual(len(rows), 4)
        with_zero = [row for row in rows if row["distractor_count"] == 0]
        with_two = [row for row in rows if row["distractor_count"] == 2]
        self.assertEqual([row["evidence_position"] for row in with_zero], ["none"])
        self.assertEqual(len({tuple(row["distractor_sources"]) for row in with_two}), 1)
        self.assertEqual(len(with_two[0]["chunks"]), 4)
        self.assertTrue(all(row["review_status"] == "needs_review" for row in rows))
        self.assertTrue(all(row["decision_evidence_indices"] is None for row in rows))
        self.assertTrue(all(row["split"] == "test" for row in rows))
        self.assertEqual(
            [row["chunks"][:2] for row in with_two if row["evidence_position"] == "first"],
            [_anchor()["chunks"]],
        )
        self.assertEqual(
            [row["chunks"][-2:] for row in with_two if row["evidence_position"] == "last"],
            [_anchor()["chunks"]],
        )

    def test_refuses_train_anchors_and_overlapping_donor_sources(self):
        with self.assertRaisesRegex(ValueError, "split='test'"):
            make_variants([_anchor(split="train")], _distractors())
        with self.assertRaisesRegex(ValueError, "distinct-source distractors"):
            make_variants(
                [_anchor()],
                _distractors(1),
                counts=(2,),
                positions=("first",),
            )
        with self.assertRaisesRegex(ValueError, "split='train'"):
            make_variants([_anchor()], [_distractors()[0] | {"split": "test"}])

    def test_bm25_ranks_lexically_relevant_chunks_first(self):
        scores = bm25_scores(
            "Orion probe Mars",
            ["A quiet ocean.", "The Orion probe reached Mars."],
        )
        self.assertGreater(scores[1], scores[0])

    def test_evaluator_requires_review_approval(self):
        rows = make_variants([_anchor()], _distractors(), counts=(0,))
        with self.assertRaisesRegex(ValueError, "review_status must be 'approved'"):
            load_reviewed_rows_from_data(rows)
        rows[0]["review_status"] = "approved"
        rows[0]["decision_evidence_indices"] = [0, 1]
        approved = load_reviewed_rows_from_data(rows)
        self.assertEqual(approved[0]["decision_evidence_indices"], [0, 1])

    def test_reports_bm25_recall_for_required_decision_evidence(self):
        class FakeScorer:
            prefilter_k = 1

            def score(self, claim, chunks):
                return 0.75

        row = _anchor()
        row.update(
            {
                "anchor_id": "test-anchor",
                "distractor_count": 0,
                "evidence_position": "none",
                "decision_evidence_indices": [0, 1],
            }
        )
        row["review_status"] = "approved"
        results = score_rows([row], FakeScorer())
        bm25_result = next(result for result in results if result["method"] == "bm25_joint")
        self.assertEqual(bm25_result["bm25_decision_evidence_recall"], 0.5)
        self.assertFalse(bm25_result["bm25_complete_decision_evidence_retained"])

    def test_metrics_and_paired_bootstrap_are_deterministic(self):
        metrics = binary_metrics([1, 0, 1], [1, 1, 0])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        rows = []
        for index, (joint, chunk_max) in enumerate(((1, 0), (1, 0), (1, 0), (0, 1))):
            for method, prediction in (("joint", joint), ("chunk_max", chunk_max)):
                rows.append(
                    {
                        "anchor_id": str(index),
                        "bootstrap_group": "large" if index < 3 else "small",
                        "method": method,
                        "label": 1,
                        "prediction": prediction,
                    }
                )
        first = paired_bootstrap_delta(rows, "joint", "chunk_max", iterations=100, seed=7)
        second = paired_bootstrap_delta(rows, "joint", "chunk_max", iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_accuracy"], 0.0)
        self.assertEqual(first["paired_groups"], 2)


def load_reviewed_rows_from_data(rows):
    from pathlib import Path
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "eval.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return load_reviewed_rows(path)


if __name__ == "__main__":
    unittest.main()
