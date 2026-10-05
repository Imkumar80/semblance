import random
import unittest

from decisionencoder.evidenceguard.data.hard_negatives import (
    build,
    flip_date,
    flip_direction,
    flip_entity,
    flip_number,
    load_lexicon,
)


class HardNegativeTests(unittest.TestCase):
    def test_number_flip_does_not_consume_temporal_numbers(self):
        self.assertIsNone(flip_number("It happened in 2024.", random.Random(0)))
        self.assertIsNone(flip_number("They met 3 years ago.", random.Random(0)))
        self.assertIsNone(flip_number("It happened on March 15, 2021.", random.Random(0)))
        self.assertIsNotNone(flip_number("The count was 1,234.", random.Random(0)))

    def test_date_flip_handles_month_year_decade_century_and_duration(self):
        cases = (
            ("It happened in March 15, 2021.", "March"),
            ("It happened in August 2019.", "August"),
            ("It began in 1970s.", "1970s"),
            ("It began in the 18th century.", "18th"),
            ("It began 3 years ago.", "3 years"),
            ("It will happen in 3 years.", "3 years"),
        )
        for claim, expected_from in cases:
            with self.subTest(claim=claim):
                result = flip_date(claim, random.Random(7))
                self.assertIsNotNone(result)
                flipped, detail = result
                self.assertNotEqual(flipped, claim)
                self.assertTrue(detail["from"].startswith(expected_from))

    def test_direction_pairs_remain_bidirectional(self):
        result = flip_direction("The number increased.", random.Random(0))
        self.assertEqual(result[1], {"from": "increased", "to": "decreased"})

    def test_regex_entity_fallback_works_without_spacy(self):
        result = flip_entity("James Wilson was elected.", {}, random.Random(0), None)
        self.assertIsNotNone(result)
        flipped, detail = result
        self.assertNotEqual(flipped, "James Wilson was elected.")
        self.assertEqual(detail["from"], "James Wilson")
        self.assertEqual(detail["entity_type"], "UNKNOWN")
        self.assertEqual(detail["method"], "regex_fallback")

    def test_build_emits_flagged_fallback_entity_rows(self):
        source = {
            "id": "train-1",
            "source": "Example",
            "premise": "James Wilson was elected in 2021.",
            "claim": "James Wilson was elected in 2021.",
            "label": 1,
            "split": "train",
        }
        rows = build([source], seed=0, use_spacy=False)
        entity_rows = [row for row in rows if row["type"] == "flip_entity"]
        self.assertEqual(len(entity_rows), 1)
        self.assertEqual(entity_rows[0]["flip_detail"]["method"], "regex_fallback")
        self.assertTrue(all(row["label"] == 0 for row in rows))

    def test_default_english_lexicon_is_external_json(self):
        lexicon = load_lexicon()
        self.assertEqual(lexicon["language"], "en")
        self.assertEqual(len(lexicon["months"]), 12)
        self.assertIn(["increased", "decreased"], lexicon["direction_pairs"])


if __name__ == "__main__":
    unittest.main()
