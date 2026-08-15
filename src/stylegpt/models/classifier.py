"""Garment classification model.

Multi-task by design: a shared backbone feeds a category head and a
color-attribute head, since a web app recommending outfits needs both
"what is this" and "what color is it" — training either head alone is
still supported (see `training/train_classifier.py`).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stylegpt.config import CLOTHING_CATEGORIES, COLOR_ATTRIBUTES
from stylegpt.models.backbone import build_backbone


class ClothingClassifier(nn.Module):
    def __init__(
        self,
        num_categories: int = len(CLOTHING_CATEGORIES),
        num_colors: int = len(COLOR_ATTRIBUTES),
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(feature_dim, num_categories)
        self.color_head = nn.Linear(feature_dim, num_colors)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        return {
            "category_logits": self.category_head(features),
            "color_logits": self.color_head(features),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, top_k: int = 3) -> list[dict]:
        """Run inference on a batch, returning human-readable top-k predictions.

        `x` is a preprocessed batch, e.g. from stacking
        `ClothingPreprocessor` outputs: shape (N, 3, H, W).
        """
        self.eval()
        outputs = self.forward(x)
        category_probs = F.softmax(outputs["category_logits"], dim=1)
        color_probs = F.softmax(outputs["color_logits"], dim=1)

        results = []
        for cat_p, col_p in zip(category_probs, color_probs):
            top_cats = torch.topk(cat_p, k=min(top_k, cat_p.numel()))
            results.append({
                "category": CLOTHING_CATEGORIES[int(top_cats.indices[0])],
                "category_confidence": float(top_cats.values[0]),
                "top_categories": [
                    {"label": CLOTHING_CATEGORIES[int(i)], "confidence": float(v)}
                    for v, i in zip(top_cats.values, top_cats.indices)
                ],
                "color": COLOR_ATTRIBUTES[int(torch.argmax(col_p))],
                "color_confidence": float(torch.max(col_p)),
            })
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu", **kwargs) -> "ClothingClassifier":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=map_location))
        model.eval()
        return model
