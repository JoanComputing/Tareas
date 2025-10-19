"""Model definitions for CREMA-D emotion recognition."""
from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def _build_mask(lengths: torch.LongTensor, max_len: int) -> torch.Tensor:
    """Return a boolean mask where ``True`` denotes padded positions."""

    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return range_row >= lengths.unsqueeze(1)


class SelfAttentionPooling(nn.Module):
    """Multi-head self-attentive pooling operating on padded sequences."""

    def __init__(
        self,
        feature_size: int,
        attention_hidden: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, attention_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, num_heads),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        """Pool a batch of sequences into a fixed-size representation."""

        batch_size, max_len, _ = sequence.shape
        attn_logits = self.projection(sequence)
        mask = _build_mask(lengths, max_len).unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)
        # (batch, heads, time) @ (batch, time, feature) -> (batch, heads, feature)
        attn_weights = attn_weights.transpose(1, 2)
        context = torch.matmul(attn_weights, sequence)
        return context.reshape(batch_size, -1)


class TemporalStatisticsPooling(nn.Module):
    """Concatenate mean and standard deviation across time."""

    def forward(self, sequence: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        batch_size, max_len, feature_size = sequence.shape
        mask = _build_mask(lengths, max_len).unsqueeze(-1)
        sequence = sequence.masked_fill(mask, 0.0)
        lengths_clamped = lengths.clamp_min(1).unsqueeze(1).to(sequence.dtype)
        mean = sequence.sum(dim=1) / lengths_clamped
        sq_sum = sequence.pow(2).sum(dim=1) / lengths_clamped
        var = torch.clamp(sq_sum - mean.pow(2), min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


class EmotionGRU(nn.Module):
    """Bidirectional GRU encoder with attentive-statistics pooling."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_hidden: int = 128,
        attention_heads: int = 4,
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
        self.attention_pool = SelfAttentionPooling(
            feature_size,
            attention_hidden=attention_hidden,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.stats_pool = TemporalStatisticsPooling()
        pooled_dim = feature_size * attention_heads + feature_size * 2
        self.norm = nn.LayerNorm(pooled_dim)
        projector_hidden = max(pooled_dim // 2, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, projector_hidden),
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
        device = features[0].device if features else lengths.device
        lengths = lengths.to(device)
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        sorted_features = [features[i] for i in sorted_indices]
        packed = pack_sequence(sorted_features, enforce_sorted=True)
        packed_outputs, _ = self.gru(packed)
        padded_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        att_pooled = self.attention_pool(padded_outputs, lengths_sorted)
        stats_pooled = self.stats_pool(padded_outputs, lengths_sorted)
        combined = torch.cat([att_pooled, stats_pooled], dim=1)
        _, unsort_indices = torch.sort(sorted_indices)
        combined = combined[unsort_indices]
        normalized = self.norm(combined)
        logits = self.classifier(normalized)
        return logits
