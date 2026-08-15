"""Training entry point for the garment category classifier.

Usage:
    python -m stylegpt.training.train_classifier --data-dir data/processed/catalog --epochs 20

Expects `--data-dir` laid out as `<data-dir>/<category>/<image>.jpg`
(see `ClothingDataset.from_folder`). Only the category head is trained
here since folder-structured data doesn't carry color labels — once a
color-labeled dataset (e.g. via `ClothingDataset.from_csv`) is
available, extend `_step` to add a weighted color-head loss term.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from stylegpt.config import DEFAULT_TRAINING_CONFIG, MODELS_DIR
from stylegpt.data.dataset import ClothingDataset
from stylegpt.data.preprocess import ClothingPreprocessor
from stylegpt.models.classifier import ClothingClassifier
from stylegpt.training.utils import get_device, set_seed


def _step(model: ClothingClassifier, images: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(images)
    loss = criterion(outputs["category_logits"], labels)
    preds = outputs["category_logits"].argmax(dim=1)
    return loss, preds


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, _ = _step(model, images, labels, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        loss, preds = _step(model, images, labels, criterion)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "classifier.pt")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--val-split", type=float, default=DEFAULT_TRAINING_CONFIG.val_split)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.seed)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    full_dataset = ClothingDataset.from_folder(args.data_dir, preprocessor=ClothingPreprocessor(mode="train"))
    if full_dataset.skipped:
        print(f"Skipped {len(full_dataset.skipped)} unreadable/missing files")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    # Validation should be deterministic, not train-time-augmented.
    val_set.dataset.preprocessor = ClothingPreprocessor(mode="eval")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=DEFAULT_TRAINING_CONFIG.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=DEFAULT_TRAINING_CONFIG.num_workers)

    model = ClothingClassifier(num_categories=len(full_dataset.class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=DEFAULT_TRAINING_CONFIG.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.output)
            print(f"  saved new best checkpoint to {args.output} (val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
