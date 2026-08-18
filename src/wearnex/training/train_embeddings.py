"""Fine-tuning entry point for the style-embedding model (metric learning).

Usage:
    python -m wearnex.training.train_embeddings --data-dir data/processed/catalog --epochs 5

`EmbeddingExtractor` already wraps Marqo-FashionCLIP, pretrained on
fashion product data -- this script is only needed if that pretrained
space needs domain-adapting to your own catalog/photos (e.g. real
user-uploaded closet photos rather than studio product shots). Only
the image tower is unfrozen: we never call `encode_text`, so the text
tower stays frozen and untouched.

`TripletCategoryDataset` samples (anchor, positive, negative) triplets
using category as a *proxy* for visual similarity: positive = same
category, negative = different category. That's a reasonable cold-start
signal but a weak one (two t-shirts can be styled very differently) —
once real style-similarity signal exists (curated outfit pairs,
co-purchase/co-view data, or human-labeled similarity judgments), swap
the sampling logic here for that instead of the category heuristic.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wearnex.config import DEFAULT_TRAINING_CONFIG, MODELS_DIR
from wearnex.data.dataset import ClothingDataset
from wearnex.models.backbone import freeze, unfreeze
from wearnex.models.embeddings import EmbeddingExtractor
from wearnex.training.utils import get_device, set_seed


class TripletCategoryDataset(Dataset):
    """Wraps a `ClothingDataset`, yielding (anchor, positive, negative) image triplets."""

    def __init__(self, base: ClothingDataset) -> None:
        self.base = base
        self.by_label: dict[int, list[int]] = {}
        for idx, (_, label) in enumerate(base.samples):
            self.by_label.setdefault(label, []).append(idx)
        # Categories with a single sample can't provide an in-class positive.
        self.labels = [label for label, idxs in self.by_label.items() if len(idxs) >= 2]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_img, anchor_label = self.base[index]
        if anchor_label not in self.by_label or len(self.by_label[anchor_label]) < 2:
            anchor_label = random.choice(self.labels)

        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.by_label[anchor_label])
        positive_img, _ = self.base[positive_idx]

        negative_label = random.choice([l for l in self.by_label if l != anchor_label])
        negative_idx = random.choice(self.by_label[negative_label])
        negative_img, _ = self.base[negative_idx]

        return anchor_img, positive_img, negative_img


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in tqdm(loader, desc="train", leave=False):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        loss = criterion(model(anchor), model(positive), model(negative))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * anchor.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "embeddings.pt")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning needs far fewer epochs than training from scratch.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=1e-5, help="Kept low to avoid destroying the pretrained embedding space.")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.seed)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    model = EmbeddingExtractor().to(device)
    freeze(model.model)
    unfreeze(model.model.visual)

    base_dataset = ClothingDataset.from_folder(args.data_dir, preprocessor=model.preprocess_image)
    triplet_dataset = TripletCategoryDataset(base_dataset)
    loader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=DEFAULT_TRAINING_CONFIG.num_workers)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=DEFAULT_TRAINING_CONFIG.weight_decay,
    )
    criterion = nn.TripletMarginLoss(margin=args.margin)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"epoch {epoch:02d}/{args.epochs} | triplet_loss={train_loss:.4f}")
        model.save(args.output)


if __name__ == "__main__":
    main()
