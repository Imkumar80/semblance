"""Train ModernBERT to classify evidence-grounded claims from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .modeling import ModernBertForGroundedness


DEFAULT_MODEL = "answerdotai/ModernBERT-base"


class ManifestDataset(Dataset):
    """Validate manifest rows and prepare context/claim tokenizer pairs."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer,
        max_length: int = 2048,
        limit: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[str, str, int]] = []
        seen_ids: set[str] = set()
        with Path(manifest_path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                where = f"{manifest_path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{where}: invalid JSON: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{where}: each manifest row must be an object")
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id.strip():
                    raise ValueError(f"{where}: id must be a non-empty string")
                if row_id in seen_ids:
                    raise ValueError(f"{where}: duplicate id {row_id!r}")
                seen_ids.add(row_id)
                claim = row.get("claim")
                if not isinstance(claim, str) or not claim.strip():
                    raise ValueError(f"{where}: claim must be non-empty text")
                chunks = row.get("chunks")
                if (
                    not isinstance(chunks, list)
                    or not chunks
                    or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
                ):
                    raise ValueError(f"{where}: chunks must be non-empty strings")
                label = row.get("label")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError(f"{where}: label must be integer 0 or 1")
                if row.get("split") != "train":
                    raise ValueError(f"{where}: trainer accepts train rows only")

                context = "\n\n".join(chunks)
                self.samples.append((f"Context: {context}", f"Claim: {claim}", label))
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"{manifest_path}: no train rows found")
        self.seen_ids = seen_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text_a, text_b, label = self.samples[index]
        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation="longest_first",
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": label,
        }


class ManifestCollator:
    """Dynamically pad tokenized sequence pairs within each batch."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
        token_features = [
            {
                key: value
                for key, value in feature.items()
                if key != "labels"
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(token_features, padding=True, return_tensors="pt")
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was selected, but CUDA is not available")
    return torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else (
        "cpu" if device_name == "auto" else device_name
    ))


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 2,
    max_length: int = 1024,
    device_name: str = "auto",
    seed: int = 0,
    limit_train_rows: int | None = None,
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or max_length < 2:
        raise ValueError("epochs, batch_size, and max_length must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if limit_train_rows is not None and limit_train_rows <= 0:
        raise ValueError("limit_train_rows must be positive")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = ManifestDataset(
        manifest_path,
        tokenizer,
        max_length=max_length,
        limit=limit_train_rows,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ManifestCollator(tokenizer),
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = ModernBertForGroundedness.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype=dtype,
    ).to(device)
    max_positions = getattr(model.config, "max_position_embeddings", max_length)
    if max_length > max_positions:
        raise ValueError(
            f"max_length={max_length} exceeds model max_position_embeddings={max_positions}"
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    epoch_losses: list[float] = []
    print(
        f"Training {len(dataset)} examples for {epochs} epoch(s) on {device}; "
        f"batch_size={batch_size}, max_length={max_length}, steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )
    for epoch in range(epochs):
        loss_sum = 0.0
        examples_seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("training produced a missing or non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_count = labels.shape[0]
            loss_sum += loss.detach().item() * batch_count
            examples_seen += batch_count
        epoch_loss = loss_sum / examples_seen
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} | loss={epoch_loss:.6f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    metadata = {
        "base_model": model_id,
        "manifest": str(manifest_path),
        "examples": len(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "attention_implementation": "sdpa",
        "device": str(device),
        "seed": seed,
        "epoch_losses": epoch_losses,
        "training_scope": (
            "smoke_test_subset"
            if limit_train_rows is not None
            else "training_manifest"
        ),
    }
    (output_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.manifest.resolve():
        parser.error("--out must not be the manifest path")
    try:
        result = train(
            args.manifest,
            args.out,
            model_id=args.model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_name=args.device,
            seed=args.seed,
            limit_train_rows=args.limit_train_rows,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
