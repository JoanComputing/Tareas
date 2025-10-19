"""Utility helpers for training and evaluation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class Metrics:
    loss: float
    accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    preds = predictions.argmax(dim=1)
    return (preds == targets).float().mean().item()


def sklearn_classification_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: List[int],
) -> Dict[str, object]:
    conf = confusion_matrix(targets, predictions, labels=labels, normalize="true")
    report = classification_report(targets, predictions, labels=labels, output_dict=True)
    return {"confusion_matrix": conf.tolist(), "classification_report": report}


class SpecAugment:
    """Lightweight SpecAugment implementation operating on mel features."""

    def __init__(
        self,
        time_mask_param: int = 12,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        p: float = 0.5,
        mel_bins: Optional[int] = None,
    ) -> None:
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.p = p
        self.mel_bins = mel_bins

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return features
        augmented = features.clone()
        num_frames, num_features = augmented.shape
        for _ in range(self.num_time_masks):
            t = int(torch.randint(low=0, high=self.time_mask_param + 1, size=(1,)).item())
            if t == 0:
                continue
            t0 = int(torch.randint(low=0, high=max(num_frames - t, 1), size=(1,)).item())
            augmented[t0 : t0 + t, :] = 0.0
        freq_dim = num_features if self.mel_bins is None else min(self.mel_bins, num_features)
        if freq_dim <= 0:
            return augmented
        for _ in range(self.num_freq_masks):
            f = int(torch.randint(low=0, high=self.freq_mask_param + 1, size=(1,)).item())
            if f == 0:
                continue
            max_start = max(freq_dim - f, 1)
            f0 = int(torch.randint(low=0, high=max_start, size=(1,)).item())
            end = min(f0 + f, freq_dim)
            augmented[:, f0:end] = 0.0
        return augmented


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
