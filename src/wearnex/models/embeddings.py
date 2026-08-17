"""Style-embedding model used for similarity-based recommendations.

Produces a fixed-size, L2-normalized vector per garment image, so that
"find similar items" reduces to nearest-neighbor search over vectors
(see `recommendation/engine.py`). Intended to be trained with a metric
learning loss (triplet/contrastive) so visually/stylistically similar
garments land close together in embedding space — plain classification
cross-entropy optimizes for a different objective and tends to produce
weaker embeddings for retrieval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wearnex.config import DEFAULT_TRAINING_CONFIG
from wearnex.models.backbone import build_backbone


class EmbeddingExtractor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = DEFAULT_TRAINING_CONFIG.embedding_dim,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Batch inference convenience: preprocessed tensor batch -> numpy embeddings."""
        self.eval()
        return self.forward(x).cpu().numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu", **kwargs) -> "EmbeddingExtractor":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=map_location))
        model.eval()
        return model
