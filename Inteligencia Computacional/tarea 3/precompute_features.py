"""Pre-compute log-mel features for the CREMA-D dataset.

This script parallelises feature extraction to speed up later training. The
resulting directory structure mirrors the original dataset splits and stores
PyTorch tensors with variable-length time dimensions to avoid zero padding.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crema_d.features import LogMelConfig, LogMelExtractor
from crema_d.utils import save_json


_EXTRACTOR: Optional[LogMelExtractor] = None


def _initialise_worker(config: Dict[str, object]) -> None:
    global _EXTRACTOR
    _EXTRACTOR = LogMelExtractor(LogMelConfig(**config))


def _process_row(
    row: Tuple[str, str, bool],
    dataset_root: Path,
    features_root: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if _EXTRACTOR is None:
        raise RuntimeError("Worker was not initialised correctly")
    relative_path, _, use_for_stats = row
    audio_path = dataset_root / relative_path
    output_path = features_root / Path(relative_path).with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features, _ = _EXTRACTOR(audio_path)
    payload = {"features": features.cpu(), "num_frames": int(features.shape[0])}
    torch.save(payload, output_path)
    if use_for_stats:
        sum_vec = features.sum(dim=0).cpu().numpy()
        sumsq_vec = (features.pow(2).sum(dim=0)).cpu().numpy()
        count = int(features.shape[0])
        return sum_vec, sumsq_vec, count
    return np.zeros(_EXTRACTOR.config.n_mels, dtype=np.float64), np.zeros(
        _EXTRACTOR.config.n_mels, dtype=np.float64
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Path to CREMA-D root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the feature tensors will be stored",
    )
    parser.add_argument("--csv", type=str, default="labels.csv", help="Relative path to labels CSV")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="Parallel workers")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--pre-emphasis", type=float, default=0.97)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = args.data_root / args.csv
    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels CSV at {labels_path}")

    df = pd.read_csv(labels_path)
    config = LogMelConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        pre_emphasis=args.pre_emphasis,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        (row.path, row.partition, row.partition == "train")
        for row in df.itertuples()
    ]

    total_sum = np.zeros(config.n_mels, dtype=np.float64)
    total_sumsq = np.zeros(config.n_mels, dtype=np.float64)
    total_frames = 0

    with mp.get_context("spawn").Pool(
        processes=args.num_workers,
        initializer=_initialise_worker,
        initargs=({
            "sample_rate": config.sample_rate,
            "n_mels": config.n_mels,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "pre_emphasis": config.pre_emphasis,
            "win_length": config.win_length,
            "center": config.center,
            "f_min": config.f_min,
            "f_max": config.f_max,
            "power": config.power,
        },),
    ) as pool:
        iterator = pool.imap_unordered(
            partial(_process_row, dataset_root=args.data_root, features_root=args.output_dir),
            rows,
        )
        for sum_vec, sumsq_vec, count in tqdm(iterator, total=len(rows), desc="Extracting"):
            total_sum += sum_vec
            total_sumsq += sumsq_vec
            total_frames += count

    if total_frames == 0:
        raise RuntimeError("No training frames processed; check the dataset partitions")

    mean = total_sum / total_frames
    variance = total_sumsq / total_frames - np.square(mean)
    std = np.sqrt(np.clip(variance, a_min=1e-8, a_max=None))
    np.savez(args.output_dir / "normalisation.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    metadata = {
        "config": {
            "sample_rate": config.sample_rate,
            "n_mels": config.n_mels,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "pre_emphasis": config.pre_emphasis,
        },
        "num_rows": len(df),
        "train_frames": int(total_frames),
    }
    save_json(metadata, args.output_dir / "metadata.json")
    print("Feature extraction complete. Normalisation statistics saved.")


if __name__ == "__main__":
    main()
