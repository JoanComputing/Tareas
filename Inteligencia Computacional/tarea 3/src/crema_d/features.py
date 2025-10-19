"""Feature extraction utilities for the CREMA-D emotion recognition task.

The design avoids zero-padding by returning variable-length time sequences.
The log-mel configuration follows common recommendations from recent speech
emotion recognition work that emphasise medium-resolution log-mel spectra with
strong pre-emphasis and energy normalisation (e.g. Triantafyllopoulos et al.,
INTERSPEECH 2023).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio


@dataclass
class LogMelConfig:
    """Configuration parameters for log-mel feature extraction."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None
    n_mels: int = 80
    f_min: float = 20.0
    f_max: Optional[float] = None
    center: bool = False
    power: float = 2.0
    pre_emphasis: float = 0.97


class LogMelExtractor:
    """Callable object that converts an audio file into log-mel features."""

    def __init__(self, config: Optional[LogMelConfig] = None) -> None:
        self.config = config or LogMelConfig()
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            center=self.config.center,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=self.config.power,
            norm="slaney",
            mel_scale="htk",
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def _pre_emphasise(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.config.pre_emphasis <= 0.0:
            return waveform
        coef = self.config.pre_emphasis
        first = waveform[..., :1]
        diff = waveform[..., 1:] - coef * waveform[..., :-1]
        return torch.cat([first, diff], dim=-1)

    def load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )
            sample_rate = self.config.sample_rate
        waveform = self._normalise_waveform(waveform)
        return waveform, sample_rate

    def _normalise_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        std = waveform.std().clamp_min(eps)
        return waveform / std

    def __call__(self, path: Path) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = self.load_audio(path)
        emphasised = self._pre_emphasise(waveform)
        spec = self._mel_transform(emphasised.unsqueeze(0))
        spec = self._to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1).contiguous()
        return spec, sample_rate


def frame_count(duration: float, config: LogMelConfig) -> int:
    """Compute number of frames for a given duration."""

    hop_length = config.hop_length
    return int(math.floor(duration * config.sample_rate / hop_length)) + 1
