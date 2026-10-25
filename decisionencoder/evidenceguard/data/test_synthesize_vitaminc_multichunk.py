import json
import random
import tempfile
import unittest
from pathlib import Path

from decisionencoder.evidenceguard.data.synthesize_vitaminc_multichunk import (
    _claims_have_distinct_predicates,
    _select_distinct_facts,
    make_examples,
)


class VitaminCMultichunkTests(unittest.TestCase):
    def test_builds_supported_and_missing_evidence_pairs_from_train_only_pages(self):
        records = []
        facts = [
            ("The probe launched from Earth.", "The probe launched aboard a rocket."),
            ("The probe reached Mars.", "The probe arrived at the planet Mars."),
            ("The probe carried a camera.", "An imaging camera was aboard the probe."),
            ("The probe transmitted radio signals.", "The spacecraft sent radio signals."),
        ]
        records.extend(
            {
                "id": f"train-{index}",
                "source": "Train Page",
                "claim": claim,
                "premise": premise,
                "label": 1,
                "split": "train",
            }
            for index, (claim, premise) in enumerate(facts)
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
        positive = next(row for row in rows if row["label"] == 1)
        self.assertTrue(
            _claims_have_distinct_predicates(
                positive["claim"].split(" Additionally, ", 1)[0],
                positive["claim"].split(" Additionally, ", 1)[1],
            )
        )
        self.assertEqual(report["candidate_train_pages"], 1)
        self.assertEqual(report["compatible_train_pages"], 1)

    def test_rejects_near_duplicate_predicates_with_conflicting_numbers_and_dates(self):
        number_claims = (
            "Serena leads the face-offs against Jelena Janković 7-3.",
            "Serena leads the face-offs against Jelena Janković 6-3.",
        )
        date_claims = (
            "The temple was completed in 1010 CE.",
            "The temple was completed in 1009 CE.",
        )
        self.assertFalse(_claims_have_distinct_predicates(*number_claims))
        self.assertFalse(_claims_have_distinct_predicates(*reversed(number_claims)))
        self.assertFalse(_claims_have_distinct_predicates(*date_claims))

    def test_generated_positive_never_pairs_conflicting_numeric_predicates(self):
        records = [
            {
                "id": "faceoffs-7-3",
                "source": "Tennis Page",
                "claim": "Serena leads the face-offs against Jelena Janković 7-3.",
                "premise": "Serena leads the face-offs against Jelena Janković 7-3.",
                "label": 1,
                "split": "train",
            },
            {
                "id": "faceoffs-6-3",
                "source": "Tennis Page",
                "claim": "Serena leads the face-offs against Jelena Janković 6-3.",
                "premise": "Serena leads the face-offs against Jelena Janković 6-3.",
                "label": 1,
                "split": "train",
            },
            {
                "id": "jankovic-professional",
                "source": "Tennis Page",
                "claim": "Jelena Janković began playing tennis professionally in 2000.",
                "premise": "Jelena Janković began playing tennis professionally in 2000.",
                "label": 1,
                "split": "train",
            },
            {
                "id": "serena-wimbledon",
                "source": "Tennis Page",
                "claim": "Serena Williams won Wimbledon in 2015.",
                "premise": "Serena Williams won Wimbledon in 2015.",
                "label": 1,
                "split": "train",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vitaminc.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            rows, _ = make_examples(path, max_rows=2, seed=3)

        positive = next(row for row in rows if row["label"] == 1)
        first_claim, second_claim = positive["claim"].split(" Additionally, ", 1)
        self.assertTrue(_claims_have_distinct_predicates(first_claim, second_claim))
        self.assertFalse(
            "7-3" in first_claim + second_claim and "6-3" in first_claim + second_claim
        )

    def test_fact_selector_rejects_conflicts_and_returns_three_distinct_predicates(self):
        facts = [
            {"claim": "The temple was completed in 1010 CE."},
            {"claim": "The temple was completed in 1009 CE."},
            {"claim": "The temple is located in Thanjavur."},
            {"claim": "The temple has an octagonal cupola."},
        ]
        selected = _select_distinct_facts(facts, random.Random(7))
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected), 3)
        self.assertTrue(
            all(
                _claims_have_distinct_predicates(selected[i]["claim"], selected[j]["claim"])
                for i in range(3)
                for j in range(i + 1, 3)
            )
        )

    def test_requires_even_output_row_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive even number"):
                make_examples(path, max_rows=3)


if __name__ == "__main__":
    unittest.main()
