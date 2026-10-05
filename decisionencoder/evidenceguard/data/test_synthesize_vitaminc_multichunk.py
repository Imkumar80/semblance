import json
import tempfile
import unittest
from pathlib import Path

from decisionencoder.evidenceguard.data.synthesize_vitaminc_multichunk import (
    make_examples,
)


class VitaminCMultichunkTests(unittest.TestCase):
    def test_builds_supported_and_missing_evidence_pairs_from_train_only_pages(self):
        records = []
        for index in range(3):
            records.append(
                {
                    "id": f"train-{index}",
                    "source": "Train Page",
                    "claim": f"Claim {index}.",
                    "premise": f"Evidence {index}.",
                    "label": 1,
                    "split": "train",
                }
            )
        records.extend(
            [
                {
                    "id": "test-page",
                    "source": "Held-out Page",
                    "claim": "Held-out claim.",
                    "premise": "Held-out evidence.",
                    "label": 1,
                    "split": "test",
                },
                {
                    "id": "leaking-train-row",
                    "source": "Held-out Page",
                    "claim": "Would leak.",
                    "premise": "Would leak.",
                    "label": 1,
                    "split": "train",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vitaminc.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            rows, report = make_examples(path, max_rows=2, seed=7)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["type"] for row in rows}, {"hop_pos", "hop_drop"})
        self.assertEqual({row["label"] for row in rows}, {0, 1})
        self.assertTrue(all(row["source"] == "Train Page" for row in rows))
        self.assertTrue(all(row["split"] == "train" for row in rows))
        self.assertTrue(all(len(row["chunks"]) == 2 for row in rows))
        self.assertEqual(report["candidate_train_pages"], 1)

    def test_requires_even_output_row_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive even number"):
                make_examples(path, max_rows=3)


if __name__ == "__main__":
    unittest.main()
