"""Feature extraction utilities for the CREMA-D emotion recognition task.

The design avoids zero-padding by returning variable-length time sequences.
The feature pipeline extends medium-resolution log-mel spectra with temporal
derivatives and pitch statistics as recommended in recent speech emotion
recognition work (e.g. Ando et al., INTERSPEECH 2022; Parthasarathy et al.,
IEEE TASLP 2020) to better capture prosodic cues without relying on 2-D CNNs.
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
    use_deltas: bool = True
    use_delta_delta: bool = True
    use_pitch: bool = True
    pitch_fmin: float = 60.0
    pitch_fmax: float = 500.0

    def __post_init__(self) -> None:
        if self.use_delta_delta and not self.use_deltas:
            raise ValueError("Delta-delta features require delta features to be enabled.")

    @property
    def hop_length_ms(self) -> float:
        return self.hop_length / self.sample_rate * 1000.0

    @property
    def frame_length_ms(self) -> float:
        win = self.win_length or self.n_fft
        return win / self.sample_rate * 1000.0


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
        self._feature_dim = self._infer_feature_dim()

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

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _infer_feature_dim(self) -> int:
        base = self.config.n_mels
        if self.config.use_deltas:
            base += self.config.n_mels
            if self.config.use_delta_delta:
                base += self.config.n_mels
        if self.config.use_pitch:
            base += 2  # pitch and NCCF from Kaldi-style pitch extractor
        return base

    def _compute_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        pitch_feats = torchaudio.functional.compute_kaldi_pitch(
            waveform.unsqueeze(0),
            sample_rate=self.config.sample_rate,
            frame_length=self.config.frame_length_ms,
            frame_shift=self.config.hop_length_ms,
            min_f0=self.config.pitch_fmin,
            max_f0=self.config.pitch_fmax,
        )
        # Output shape: (num_frames, 2) -> pitch frequency and NCCF
        return torch.nan_to_num(pitch_feats.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_deltas(self, spec: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # spec is (time, mel)
        mel_time = spec.transpose(0, 1).unsqueeze(0)
        delta_time = torchaudio.functional.compute_deltas(mel_time)
        delta = delta_time.squeeze(0).transpose(0, 1)
        delta2 = None
        if self.config.use_delta_delta:
            delta2 = torchaudio.functional.compute_deltas(delta_time).squeeze(0).transpose(0, 1)
        return delta, delta2

    def __call__(self, path: Path) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = self.load_audio(path)
        emphasised = self._pre_emphasise(waveform)
        spec = self._mel_transform(emphasised.unsqueeze(0))
        spec = self._to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1).contiguous()

        features = [spec]
        if self.config.use_deltas:
            delta, delta2 = self._compute_deltas(spec)
            features.append(delta)
            if self.config.use_delta_delta and delta2 is not None:
                features.append(delta2)
        target_len = min(f.size(0) for f in features)
        features = [f[:target_len] for f in features]
        if self.config.use_pitch:
            pitch = self._compute_pitch(emphasised).to(spec.dtype)
            if pitch.ndim == 1:
                pitch = pitch.unsqueeze(1)
            if pitch.size(1) == 1:
                pitch = torch.cat([pitch, torch.zeros_like(pitch)], dim=1)
            if pitch.size(0) < target_len:
                pad = torch.zeros(target_len - pitch.size(0), pitch.size(1), dtype=pitch.dtype)
                pitch = torch.cat([pitch, pad], dim=0)
            elif pitch.size(0) > target_len:
                pitch = pitch[:target_len]
            features.append(pitch)

        combined = torch.cat(features, dim=1).contiguous()
        return combined, sample_rate


def frame_count(duration: float, config: LogMelConfig) -> int:
    """Compute number of frames for a given duration."""

    hop_length = config.hop_length
    return int(math.floor(duration * config.sample_rate / hop_length)) + 1
