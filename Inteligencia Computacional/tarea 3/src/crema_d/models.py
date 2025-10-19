"""Model definitions for CREMA-D emotion recognition."""
from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional embeddings (Vaswani et al., 2017)."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build_pe(max_len))
        self.dropout = nn.Dropout(dropout)

    def _build_pe(self, length: int) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / self.dim)
        )
        pe = torch.zeros(length, self.dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _maybe_extend(self, seq_len: int) -> None:
        if seq_len <= self.pe.size(0):
            return
        new_pe = self._build_pe(seq_len)
        self.pe = new_pe.to(self.pe.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to an input of shape (batch, time, dim)."""

        self._maybe_extend(x.size(1))
        pos = self.pe[: x.size(1)]
        x = x + pos.unsqueeze(0)
        return self.dropout(x)


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
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 1024,
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
        self.use_transformer = transformer_layers > 0
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_size,
                nhead=transformer_heads,
                dim_feedforward=transformer_ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.positional = SinusoidalPositionalEncoding(feature_size, dropout=dropout)
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
        if self.use_transformer:
            mask = _build_mask(lengths_sorted, padded_outputs.size(1))
            transformed = self.positional(padded_outputs)
            transformed = self.transformer(transformed, src_key_padding_mask=mask)
        else:
            transformed = padded_outputs
        att_pooled = self.attention_pool(transformed, lengths_sorted)
        stats_pooled = self.stats_pool(transformed, lengths_sorted)
        combined = torch.cat([att_pooled, stats_pooled], dim=1)
        _, unsort_indices = torch.sort(sorted_indices)
        combined = combined[unsort_indices]
        normalized = self.norm(combined)
        logits = self.classifier(normalized)
        return logits
