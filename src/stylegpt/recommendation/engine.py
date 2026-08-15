"""Nearest-neighbor recommendation index over garment embeddings.

Two recommendation modes:
  - `recommend_similar`: nearest neighbors in embedding space — "more
    like this item".
  - `recommend_complementary`: nearest neighbors *within* the set of
    categories that conventionally pair with the query item's category
    — "what goes with this item". See `compatibility.py`.

The index is built once (offline, via `scripts/build_index.py`) from a
catalog of preprocessed + embedded items, then loaded read-only by the
inference/API layer at request time.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from stylegpt.recommendation.compatibility import complementary_categories

REQUIRED_METADATA_COLUMNS = {"item_id", "category", "image_path"}


class RecommendationEngine:
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.embeddings: np.ndarray | None = None
        self.metadata: pd.DataFrame | None = None
        self._index: NearestNeighbors | None = None

    def build_index(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        """Fit the index over a catalog's embeddings + metadata.

        `embeddings[i]` must correspond to `metadata.iloc[i]`; metadata
        must contain at least `REQUIRED_METADATA_COLUMNS`.
        """
        missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
        if missing:
            raise ValueError(f"metadata is missing required columns: {missing}")
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            )

        self.embeddings = embeddings
        self.metadata = metadata.reset_index(drop=True)
        self._index = NearestNeighbors(metric=self.metric).fit(embeddings)

    def _require_index(self) -> None:
        if self._index is None:
            raise RuntimeError("Index not built — call build_index() or load() first")

    def recommend_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_item_id: str | None = None,
    ) -> pd.DataFrame:
        """Top-k catalog items closest to `query_embedding`."""
        self._require_index()
        query = np.asarray(query_embedding).reshape(1, -1)
        fetch_k = k + 1 if exclude_item_id is not None else k
        fetch_k = min(fetch_k, len(self.metadata))

        distances, indices = self._index.kneighbors(query, n_neighbors=fetch_k)
        results = self.metadata.iloc[indices[0]].copy()
        results["distance"] = distances[0]

        if exclude_item_id is not None:
            results = results[results["item_id"] != exclude_item_id]
        return results.head(k).reset_index(drop=True)

    def recommend_complementary(
        self,
        item_id: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """Top-k items whose category pairs well with `item_id`'s category.

        Ranks candidates within the compatible-category subset by
        embedding distance to the query item, so suggestions are still
        stylistically close, not just category-matched at random.
        """
        self._require_index()
        matches = self.metadata.index[self.metadata["item_id"] == item_id]
        if len(matches) == 0:
            raise KeyError(f"Unknown item_id: {item_id}")
        query_idx = matches[0]

        target_categories = complementary_categories(self.metadata.loc[query_idx, "category"])
        candidate_mask = self.metadata["category"].isin(target_categories)
        if not candidate_mask.any():
            return self.metadata.iloc[0:0].copy()

        candidate_embeddings = self.embeddings[candidate_mask.to_numpy()]
        candidate_metadata = self.metadata[candidate_mask].reset_index(drop=True)

        local_index = NearestNeighbors(metric=self.metric).fit(candidate_embeddings)
        query = self.embeddings[query_idx].reshape(1, -1)
        fetch_k = min(k, len(candidate_metadata))
        distances, indices = local_index.kneighbors(query, n_neighbors=fetch_k)

        results = candidate_metadata.iloc[indices[0]].copy()
        results["distance"] = distances[0]
        return results.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self._require_index()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata, "metric": self.metric}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RecommendationEngine":
        with open(path, "rb") as f:
            state = pickle.load(f)
        engine = cls(metric=state["metric"])
        engine.build_index(state["embeddings"], state["metadata"])
        return engine
