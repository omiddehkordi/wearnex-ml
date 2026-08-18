"""Style-embedding model used for similarity-based recommendations.

Produces a fixed-size, L2-normalized vector per garment image, so that
"find similar items" reduces to nearest-neighbor search over vectors
(see `recommendation/engine.py`). Wraps Marqo-FashionCLIP, a CLIP model
fine-tuned on fashion product data, so garments land in a style-aware
embedding space without any training of our own.

Loaded via `open_clip` rather than `transformers`: the transformers
checkpoint for this model requires `trust_remote_code=True`, which
executes code bundled in the model repo at load time. `open_clip`
loads the same published weights into its own model code instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from wearnex.data.preprocess import ClothingPreprocessor

MARQO_FASHIONCLIP = "hf-hub:Marqo/marqo-fashionCLIP"


class EmbeddingExtractor(nn.Module):
    def __init__(self, model_name: str = MARQO_FASHIONCLIP) -> None:
        super().__init__()
        model, _, clip_transform = open_clip.create_model_and_transforms(model_name)
        self.model = model
        self.embedding_dim = model.visual.output_dim
        self._clip_transform = clip_transform
        self._cleanup = ClothingPreprocessor(mode="eval")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> a single CLIP-ready tensor (not yet batched).

        Runs the shared garment cleanup (crop-to-subject, denoise,
        contrast boost) before handing off to CLIP's own resize and
        normalization, since this backbone expects different input
        statistics than the ImageNet-normalized tensors
        `ClothingPreprocessor.__call__` produces for the classifier.
        """
        image = self._cleanup.preprocess_image(image)
        return self._clip_transform(image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x)
        return F.normalize(features, p=2, dim=1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Batch inference convenience: preprocessed tensor batch -> numpy embeddings."""
        self.eval()
        return self.forward(x).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Convenience: raw PIL images -> numpy embeddings (preprocessing + batching included)."""
        device = next(self.parameters()).device
        batch = torch.stack([self.preprocess_image(img) for img in images]).to(device)
        return self.encode(batch)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu", **kwargs) -> "EmbeddingExtractor":
        extractor = cls(**kwargs)
        extractor.model.load_state_dict(torch.load(path, map_location=map_location))
        extractor.eval()
        return extractor
