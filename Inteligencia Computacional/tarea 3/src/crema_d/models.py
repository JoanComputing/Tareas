"""Model definitions for CREMA-D emotion recognition."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence


class EmotionGRU(nn.Module):
    """Bidirectional GRU encoder with projection head."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feature_size = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feature_size)
        projector_hidden = max(feature_size // 2, 64)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, projector_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden, num_classes),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        lengths: torch.LongTensor,
    ) -> torch.Tensor:
        if len(features) != lengths.numel():
            raise ValueError("Mismatch between features and lengths")
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        sorted_features = [features[i] for i in sorted_indices]
        packed = pack_sequence(sorted_features, enforce_sorted=True)
        packed_outputs, hidden = self.gru(packed)
        if self.bidirectional:
            last_hidden = torch.cat(
                [hidden[-2], hidden[-1]],
                dim=1,
            )
        else:
            last_hidden = hidden[-1]
        _, unsort_indices = torch.sort(sorted_indices)
        last_hidden = last_hidden[unsort_indices]
        normalized = self.norm(last_hidden)
        logits = self.classifier(normalized)
        return logits
