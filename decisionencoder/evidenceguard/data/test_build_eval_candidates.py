import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

from decisionencoder.evidenceguard.data.build_eval_candidates import (
    build_candidate_splits,
    main as build_candidates_main,
)


def _row(index, label, source=None):
    return {
        "id": f"claim-{index}",
        "source": source or f"source-{index}",
        "claim": f"Claim number {index}",
        "chunks": [f"Context for claim {index}."],
        "label": label,
        "split": "test",
    }


class BuildEvalCandidateTests(unittest.TestCase):
    def test_creates_balanced_reproducible_source_disjoint_splits_excluding_pilot(self):
        anchors = [_row(index, index % 2) for index in range(80)]
        pilot = [{"anchor_id": "claim-0", "source": "source-0"}]

        dev, test = build_candidate_splits(
            anchors,
            pilot,
            anchors_per_split=20,
            unsupported_per_split=10,
            seed=27,
        )
        dev_again, test_again = build_candidate_splits(
            anchors,
            pilot,
            anchors_per_split=20,
            unsupported_per_split=10,
            seed=27,
        )

        self.assertEqual(dev, dev_again)
        self.assertEqual(test, test_again)
        self.assertEqual(len(dev), 20)
        self.assertEqual(len(test), 20)
        self.assertEqual(sum(row["label"] == 0 for row in dev), 10)
        self.assertEqual(sum(row["label"] == 0 for row in test), 10)
        self.assertFalse(
            {row["source"].casefold() for row in dev}
            & {row["source"].casefold() for row in test}
        )
        self.assertNotIn("source-0", {row["source"].casefold() for row in dev + test})
        self.assertTrue(
            all(row["review_status"] == "needs_review" for row in dev + test)
        )
        self.assertTrue(
            all(row["decision_evidence_indices"] is None for row in dev + test)
        )

    def test_fails_if_there_are_not_enough_distinct_sources(self):
        anchors = [_row(0, 0, "shared"), _row(1, 1, "shared")]
        with self.assertRaisesRegex(ValueError, "distinct sources for label 0"):
            build_candidate_splits(
                anchors,
                [],
                anchors_per_split=2,
                unsupported_per_split=1,
            )

    def test_development_only_cli_does_not_write_locked_test(self):
        anchors = [_row(index, index % 2) for index in range(80)]
        pilot = [{"anchor_id": "claim-0", "source": "source-0"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors_path = root / "anchors.jsonl"
            pilot_path = root / "pilot.jsonl"
            out_dir = root / "candidates"
            anchors_path.write_text(
                "".join(json.dumps(row) + "\n" for row in anchors),
                encoding="utf-8",
            )
            pilot_path.write_text(
                "".join(json.dumps(row) + "\n" for row in pilot),
                encoding="utf-8",
            )
            argv = [
                "build_eval_candidates",
                "--anchors",
                str(anchors_path),
                "--pilot-manifest",
                str(pilot_path),
                "--out-dir",
                str(out_dir),
                "--anchors-per-split",
                "20",
                "--unsupported-per-split",
                "10",
                "--split",
                "development",
            ]
            with patch.object(sys, "argv", argv):
                build_candidates_main()
            self.assertTrue((out_dir / "development_needs_review.jsonl").exists())
            self.assertFalse((out_dir / "locked_test_needs_review.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
