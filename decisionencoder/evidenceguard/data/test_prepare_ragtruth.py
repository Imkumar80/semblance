import json
import tempfile
import unittest
from pathlib import Path

from decisionencoder.evidenceguard.data.prepare_ragtruth import prepare_ragtruth


class PrepareRAGTruthTests(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixtures(self, root):
        responses_path = root / "responses.jsonl"
        sources_path = root / "sources.jsonl"
        self._write_jsonl(
            responses_path,
            [
                {
                    "id": "train-1",
                    "source_id": "source-shared",
                    "split": "train",
                    "quality": "good",
                    "response": "Unsupported assertion.",
                    "labels": [{"start": 0, "end": 11, "text": "Unsupported"}],
                },
                {
                    "id": "test-1",
                    "source_id": "source-shared",
                    "split": "test",
                    "quality": "good",
                    "response": "Held-out answer.",
                    "labels": [],
                },
                {
                    "id": "train-2",
                    "source_id": "source-train",
                    "split": "train",
                    "quality": "good",
                    "response": "Grounded answer.",
                    "labels": [],
                },
                {
                    "id": "train-3",
                    "source_id": "source-train",
                    "split": "train",
                    "quality": "truncated",
                    "response": "Truncated.",
                    "labels": [],
                },
            ],
        )
        self._write_jsonl(
            sources_path,
            [
                {
                    "source_id": "source-shared",
                    "task_type": "QA",
                    "source": "Test corpus",
                    "source_info": {
                        "question": "Question?",
                        "passages": "passage 1: First evidence.\n\npassage 2: Second evidence.",
                    },
                },
                {
                    "source_id": "source-train",
                    "task_type": "Summary",
                    "source": "Train corpus",
                    "source_info": "Article content.",
                },
            ],
        )
        return responses_path, sources_path

    def test_requires_explicit_local_rights_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            responses, sources = self._fixtures(root)
            with self.assertRaisesRegex(ValueError, "source-document rights"):
                prepare_ragtruth(
                    responses,
                    sources,
                    root / "train.jsonl",
                    root / "test.jsonl",
                )

    def test_outputs_official_splits_and_excludes_cross_split_source_from_train(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            responses, sources = self._fixtures(root)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            report = prepare_ragtruth(
                responses,
                sources,
                train_path,
                test_path,
                confirm_local_training_rights=True,
            )
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            test_rows = [
                json.loads(line)
                for line in test_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["id"] for row in train_rows], ["ragtruth-response-train-2"])
        self.assertEqual([row["id"] for row in test_rows], ["ragtruth-response-test-1"])
        self.assertEqual(train_rows[0]["label"], 1)
        self.assertEqual(test_rows[0]["label"], 1)
        self.assertEqual(len(test_rows[0]["chunks"]), 2)
        self.assertEqual(report["train_rows_removed_cross_split_source"], 1)
        self.assertEqual(report["quality_rows_removed"]["train"], 1)
        self.assertEqual(report["cross_split_source_groups"], 1)


if __name__ == "__main__":
    unittest.main()
