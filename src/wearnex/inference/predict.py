"""Single entry point tying preprocessing + models + recommendation index together.

This is what the web API (and any offline scripts) should import rather
than wiring up the classifier, embedder, and recommendation engine
separately — it keeps "how a raw image becomes a classification +
recommendations" defined in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from wearnex.config import MODELS_DIR
from wearnex.data.preprocess import ClothingPreprocessor
from wearnex.models.classifier import ClothingClassifier
from wearnex.models.embeddings import EmbeddingExtractor
from wearnex.recommendation.engine import RecommendationEngine
from wearnex.training.utils import get_device


class InferencePipeline:
    def __init__(
        self,
        classifier_path: str | Path = MODELS_DIR / "classifier.pt",
        recommendation_index_path: str | Path = MODELS_DIR / "recommendation_index.pkl",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.preprocessor = ClothingPreprocessor(mode="eval")

        self.classifier = ClothingClassifier.load(classifier_path, map_location=str(self.device)).to(self.device)
        self.embedder = EmbeddingExtractor().to(self.device)
        self.recommender = RecommendationEngine.load(recommendation_index_path)

    def _to_batch(self, image: Image.Image) -> torch.Tensor:
        return self.preprocessor(image).unsqueeze(0).to(self.device)

    def classify(self, image: Image.Image) -> dict:
        batch = self._to_batch(image)
        return self.classifier.predict(batch, top_k=3)[0]

    def embed(self, image: Image.Image):
        return self.embedder.encode_images([image])[0]

    def recommend_similar(self, image: Image.Image, k: int = 10) -> pd.DataFrame:
        embedding = self.embed(image)
        return self.recommender.recommend_similar(embedding, k=k)

    def recommend_complementary(self, item_id: str, k: int = 10) -> pd.DataFrame:
        return self.recommender.recommend_complementary(item_id, k=k)

    def analyze(self, image: Image.Image, k: int = 10) -> dict:
        """Full pipeline for a single uploaded image: classification + similar items.

        This is the shape the web app's "upload a photo" endpoint should
        return (see `api/main.py`).
        """
        classification = self.classify(image)
        embedding = self.embed(image)
        similar = self.recommender.recommend_similar(embedding, k=k)
        return {
            "classification": classification,
            "similar_items": similar.to_dict(orient="records"),
        }
