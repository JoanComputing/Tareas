"""Training script for CREMA-D speech emotion recognition.

Usage example (after downloading the dataset in /content/CREMA-D)::

    python precompute_features.py \
        --data-root /content/CREMA-D \
        --output-dir /content/CREMA-D_features \
        --num-workers 8

    python train_crema_d.py \
        --data-root /content/CREMA-D \
        --features-root /content/CREMA-D_features \
        --output-dir /content/experiments/crema_d \
        --epochs 80 \
        --batch-size 16

The training pipeline follows best practices reported in recent speech emotion
recognition studies that rely on recurrent encoders and log-mel inputs while
avoiding zero-padding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crema_d.data import EmotionDataset, LabelEncoder, emotion_collate, stratified_split
from crema_d.models import EmotionGRU
from crema_d.plotting import plot_confusion_matrix, plot_history
from crema_d.utils import Metrics, SpecAugment, set_seed, sklearn_classification_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--features-root", type=Path, required=True, help="Directory with pre-computed features")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to store models and plots")
    parser.add_argument("--csv", type=str, default="labels.csv", help="Relative path to labels CSV")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-hidden", type=int, default=160)
    parser.add_argument("--use-spec-augment", action="store_true")
    parser.add_argument("--no-spec-augment", dest="use_spec_augment", action="store_false")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.set_defaults(use_spec_augment=True, use_class_weights=True)
    return parser.parse_args()


def load_normalisation_stats(features_root: Path) -> Dict[str, np.ndarray]:
    stats_path = features_root / "normalisation.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalisation statistics not found at {stats_path}. Run precompute_features.py first."
        )
    data = np.load(stats_path)
    return {"mean": data["mean"], "std": data["std"]}


def make_dataloaders(
    df: pd.DataFrame,
    features_root: Path,
    encoder: LabelEncoder,
    normalisation_stats: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    use_spec_augment: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    augment_fn = SpecAugment(p=0.4) if use_spec_augment else None
    train_df = stratified_split(df, "train")
    val_df = stratified_split(df, "validation")
    test_df = stratified_split(df, "test")

    train_dataset = EmotionDataset(train_df, features_root, encoder, normalisation_stats, augment_fn)
    val_dataset = EmotionDataset(val_df, features_root, encoder, normalisation_stats)
    test_dataset = EmotionDataset(test_df, features_root, encoder, normalisation_stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=emotion_collate,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=emotion_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=emotion_collate,
        pin_memory=True,
    )
    encoded = train_df["class"].map(encoder.class_to_index).astype(int)
    counts = (
        encoded.value_counts().reindex(range(encoder.num_classes), fill_value=0).sort_index()
    )
    return train_loader, eval_loader, test_loader, counts.to_numpy(dtype=np.int64)


def compute_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    class_counts = np.maximum(class_counts, 1)
    total = class_counts.sum()
    weights = total / (class_counts * class_counts.size)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 0.0,
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, targets, lengths in data_loader:
        features = [f.to(device) for f in features]
        targets = targets.to(device)
        lengths = lengths.to(device)
        logits = model(features, lengths)
        loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return Metrics(loss=avg_loss, accuracy=avg_acc)


def evaluate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for features, labels, lengths in data_loader:
            features = [f.to(device) for f in features]
            lengths = lengths.to(device)
            logits = model(features, lengths)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            targets.extend(labels.numpy().tolist())
    return np.array(predictions), np.array(targets)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.data_root / args.csv
    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels CSV at {labels_path}")

    df = pd.read_csv(labels_path)
    encoder = LabelEncoder.from_series(df["class"])
    normalisation_stats = load_normalisation_stats(args.features_root)

    train_loader, val_loader, test_loader, class_counts = make_dataloaders(
        df,
        args.features_root,
        encoder,
        normalisation_stats,
        args.batch_size,
        args.num_workers,
        args.use_spec_augment,
    )

    model = EmotionGRU(
        input_size=normalisation_stats["mean"].shape[0],
        num_classes=encoder.num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        attention_hidden=args.attention_hidden,
    ).to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(class_counts).to(device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(train_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_accuracy"].append(val_metrics.accuracy)

        print(
            f"Epoch {epoch:03d} | Train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.3f} | "
            f"Val loss {val_metrics.loss:.4f} acc {val_metrics.accuracy:.3f}"
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_accuracy": best_val_acc,
            }
            torch.save(best_state, args.output_dir / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce any model checkpoint")

    model.load_state_dict(best_state["model"])
    predictions, targets = evaluate_predictions(model, test_loader, device)
    summary = sklearn_classification_summary(predictions, targets, list(range(encoder.num_classes)))
    test_accuracy = (predictions == targets).mean()
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Persist artefacts
    history_path = args.output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    plot_history(history, args.output_dir / "training_curves.png")
    matrix = np.array(summary["confusion_matrix"])
    class_names = [str(encoder.decode(i)) for i in range(encoder.num_classes)]
    plot_confusion_matrix(matrix, class_names, args.output_dir / "confusion_matrix.png")

    report_path = args.output_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary["classification_report"], fp, indent=2)

    metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump({"test_accuracy": float(test_accuracy)}, fp, indent=2)

    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
