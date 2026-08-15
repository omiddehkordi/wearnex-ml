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

from stylegpt.config import MODELS_DIR
from stylegpt.data.preprocess import ClothingPreprocessor
from stylegpt.models.classifier import ClothingClassifier
from stylegpt.models.embeddings import EmbeddingExtractor
from stylegpt.recommendation.engine import RecommendationEngine
from stylegpt.training.utils import get_device


class InferencePipeline:
    def __init__(
        self,
        classifier_path: str | Path = MODELS_DIR / "classifier.pt",
        embedding_model_path: str | Path = MODELS_DIR / "embeddings.pt",
        recommendation_index_path: str | Path = MODELS_DIR / "recommendation_index.pkl",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.preprocessor = ClothingPreprocessor(mode="eval")

        self.classifier = ClothingClassifier.load(classifier_path, map_location=str(self.device)).to(self.device)
        self.embedder = EmbeddingExtractor.load(embedding_model_path, map_location=str(self.device)).to(self.device)
        self.recommender = RecommendationEngine.load(recommendation_index_path)

    def _to_batch(self, image: Image.Image) -> torch.Tensor:
        return self.preprocessor(image).unsqueeze(0).to(self.device)

    def classify(self, image: Image.Image) -> dict:
        batch = self._to_batch(image)
        return self.classifier.predict(batch, top_k=3)[0]

    def embed(self, image: Image.Image):
        batch = self._to_batch(image)
        return self.embedder.encode(batch)[0]

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
