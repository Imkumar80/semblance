import json
import tempfile
import unittest
from pathlib import Path

from decisionencoder.evidenceguard.data.prepare_train_component import (
    prepare_train_component,
)


class PrepareTrainComponentTests(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_filters_by_id_claim_and_source_and_keeps_only_train(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "hard-negatives.jsonl"
            evaluation_path = root / "evaluation.jsonl"
            output_path = root / "hard-negatives-train.jsonl"
            self._write_jsonl(
                input_path,
                [
                    {
                        "id": "train-source-hit",
                        "source": "Page One",
                        "claim": "A distinct claim.",
                        "label": 0,
                        "split": "train",
                    },
                    {
                        "id": "train-claim-hit",
                        "source": "Page Two",
                        "claim": "Same claim!",
                        "label": 0,
                        "split": "train",
                    },
                    {
                        "id": "train-safe",
                        "source": "Page Three",
                        "claim": "Safe claim.",
                        "label": 0,
                        "split": "train",
                    },
                    {
                        "id": "test-ignored",
                        "source": "Test Page",
                        "claim": "Test claim.",
                        "label": 0,
                        "split": "test",
                    },
                ],
            )
            self._write_jsonl(
                evaluation_path,
                [
                    {
                        "id": "eval-1",
                        "source": "Page One",
                        "claim": "Evaluation claim.",
                        "split": "test",
                    },
                    {
                        "id": "eval-2",
                        "source": "Page Four",
                        "claim": "Same claim.",
                        "split": "val",
                    },
                ],
            )
            report = prepare_train_component(
                input_path, [evaluation_path], output_path
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["id"] for row in rows], ["train-safe"])
        self.assertEqual(report["train_rows_removed"], 2)
        self.assertEqual(
            report["removed_for_overlap"],
            {"id": 0, "claim": 1, "source": 1},
        )
        self.assertEqual(report["non_train_rows_ignored"], 1)


if __name__ == "__main__":
    unittest.main()
