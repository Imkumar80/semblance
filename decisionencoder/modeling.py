"""ModernBERT binary sequence classifier for evidence groundedness."""

from __future__ import annotations

import torch
from torch import nn
from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ModernBertForGroundedness(ModernBertPreTrainedModel):
    """Classify whether a claim is supported by its packed evidence context."""

    base_model_prefix = "model"

    def __init__(self, config) -> None:
        config.num_labels = 1
        super().__init__(config)
        self.num_labels = 1
        self.model = ModernBertModel(config)
        dropout_rate = (
            config.classifier_dropout
            or getattr(config, "hidden_dropout_prob", None)
            or 0.1
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(config.hidden_size, 1),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.to(dtype=logits.dtype)
            )

        if not return_dict:
            output = (logits,) + outputs.to_tuple()[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
