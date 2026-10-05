import json
import tempfile
import unittest
from pathlib import Path

from decisionencoder.evidenceguard.data.build_train_manifest import (
    compose_manifest,
    write_manifest,
)


class TrainManifestTests(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixture(self, root, overlap=False):
        counts = {"hard_negatives": 6, "multichunk": 7, "ragtruth_train": 7}
        paths = {}
        for component, count in counts.items():
            rows = []
            for index in range(count):
                source = (
                    "eval-source"
                    if overlap and component == "hard_negatives" and index == 0
                    else f"{component}-source-{index}"
                )
                rows.append(
                    {
                        "id": f"{component}-{index}",
                        "source": source,
                        "claim": f"Claim from {component} number {index}.",
                        "chunks": (
                            ["Evidence text one.", "Evidence text two."]
                            if component == "multichunk"
                            else ["Evidence text."]
                        ),
                        "label": index % 2,
                        "split": "train",
                    }
                )
            rows.append({"split": "test"})
            path = root / f"{component}.jsonl"
            self._write_jsonl(path, rows)
            paths[component] = path

        evaluation_path = root / "evaluation.jsonl"
        self._write_jsonl(
            evaluation_path,
            [
                {
                    "id": "eval-1",
                    "source": "eval-source",
                    "claim": "Evaluation claim.",
                    "split": "test",
                }
            ],
        )
        return paths, [evaluation_path]

    def test_uses_exact_requested_shares_and_train_rows_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs, evaluations = self._fixture(root)
            rows, report = compose_manifest(inputs, evaluations, seed=8)
            output = root / "train_manifest.jsonl"
            write_manifest(rows, output)
            written_rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 20)
        self.assertEqual(len(written_rows), 20)
        self.assertEqual(report["selected_rows"], {
            "hard_negatives": 6,
            "multichunk": 7,
            "ragtruth_train": 7,
        })
        self.assertTrue(all(row["split"] == "train" for row in rows))
        self.assertEqual(len({row["id"] for row in rows}), 20)
        self.assertEqual(
            {row["source_component"] for row in rows},
            {"hard_negatives", "multichunk", "ragtruth_train"},
        )

    def test_fails_if_any_training_source_overlaps_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs, evaluations = self._fixture(root, overlap=True)
            with self.assertRaisesRegex(ValueError, "source overlaps an evaluation source"):
                compose_manifest(inputs, evaluations, total_rows=20)


if __name__ == "__main__":
    unittest.main()
