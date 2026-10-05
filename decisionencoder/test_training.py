import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import ModernBertConfig

from decisionencoder.modeling import ModernBertForGroundedness
from decisionencoder.train import ManifestCollator, ManifestDataset


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text_a, text_b, **kwargs):
        max_length = kwargs["max_length"]
        tokens = [1] + [2] * min(len((text_a + text_b).split()), max_length - 2) + [3]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def pad(self, features, **kwargs):
        if kwargs.get("return_tensors") != "pt":
            raise ValueError("expected PyTorch tensor output")
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        for feature in features:
            padding_length = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [0] * padding_length)
            attention_masks.append(feature["attention_mask"] + [0] * padding_length)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
        }


class TrainingTests(unittest.TestCase):
    def test_dataset_packs_chunks_and_collates_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "train-1",
                        "chunks": ["First evidence.", "Second evidence."],
                        "claim": "Combined claim.",
                        "label": 1,
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tokenizer = TinyTokenizer()
            dataset = ManifestDataset(manifest, tokenizer, max_length=16)
            batch = ManifestCollator(tokenizer)([dataset[0]])

        self.assertEqual(len(dataset), 1)
        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.assertEqual(batch["labels"].tolist(), [1.0])
        self.assertEqual(dataset.samples[0][0], "Context: First evidence.\n\nSecond evidence.")

    def test_limited_smoke_mode_still_validates_all_manifest_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.jsonl"
            train_row = {
                "id": "train-1",
                "chunks": ["Evidence."],
                "claim": "Claim.",
                "label": 1,
                "split": "train",
            }
            eval_row = {**train_row, "id": "test-1", "split": "test"}
            manifest.write_text(
                json.dumps(train_row) + "\n" + json.dumps(eval_row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "train rows only"):
                ManifestDataset(manifest, TinyTokenizer(), limit=1)

    def test_model_forward_computes_binary_loss_and_backward(self):
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            local_attention=16,
            classifier_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            cls_token_id=1,
            eos_token_id=2,
            sep_token_id=2,
        )
        model = ModernBertForGroundedness(config)
        outputs = model(
            input_ids=torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            labels=torch.tensor([1.0, 0.0]),
        )
        self.assertEqual(outputs.logits.shape, (2,))
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        self.assertIsNotNone(model.classifier[0].weight.grad)

    def test_model_supports_tuple_return(self):
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            local_attention=16,
            classifier_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            cls_token_id=1,
            eos_token_id=2,
            sep_token_id=2,
        )
        model = ModernBertForGroundedness(config)
        output = model(input_ids=torch.tensor([[1, 2, 3]]), return_dict=False)
        self.assertEqual(output[0].shape, (1,))


if __name__ == "__main__":
    unittest.main()
