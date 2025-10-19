"""Dataset utilities for CREMA-D.

The dataset returns variable-length sequences so that the model can ingest
packed sequences without zero-padding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class LabelEncoder:
    """Simple label encoder for mapping raw labels to contiguous ids."""

    classes_: List[int]
    class_to_index: Dict[int, int]

    @classmethod
    def from_series(cls, series: pd.Series) -> "LabelEncoder":
        unique = sorted(int(x) for x in series.unique())
        mapping = {label: idx for idx, label in enumerate(unique)}
        return cls(unique, mapping)

    def encode(self, label: int) -> int:
        return self.class_to_index[int(label)]

    def decode(self, index: int) -> int:
        return self.classes_[index]

    @property
    def num_classes(self) -> int:
        return len(self.classes_)


class EmotionDataset(Dataset):
    """Dataset that loads pre-computed log-mel features from disk."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features_root: Path,
        encoder: LabelEncoder,
        normalisation_stats: Dict[str, np.ndarray],
        augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.features_root = features_root
        self.encoder = encoder
        self.augment_fn = augment_fn
        mean = normalisation_stats["mean"].astype(np.float32)
        std = normalisation_stats["std"].astype(np.float32)
        std[std < 1e-6] = 1e-6
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_features(self, rel_path: str) -> torch.Tensor:
        feature_path = self.features_root / Path(rel_path).with_suffix(".pt")
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features at {feature_path}")
        payload = torch.load(feature_path)
        features: torch.Tensor = payload["features"].float()
        return features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        features = self._load_features(row["path"])
        features = (features - self.mean) / self.std
        if self.augment_fn is not None:
            features = self.augment_fn(features)
        label = self.encoder.encode(int(row["class"]))
        return features, label


def emotion_collate(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.LongTensor]:
    """Collate function returning a list of feature sequences.

    The function keeps variable-length sequences intact while collecting the
    lengths for later use in packed sequences.
    """

    features, labels = zip(*batch)
    lengths = torch.tensor([feat.size(0) for feat in features], dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return list(features), labels_tensor, lengths


def stratified_split(
    dataframe: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    """Return rows belonging to a given partition."""

    filtered = dataframe[dataframe["partition"] == partition].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for partition '{partition}'")
    return filtered
