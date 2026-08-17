"""Build the recommendation index (embeddings + metadata) from a catalog of images.

Usage:
    python scripts/build_index.py --data-dir data/processed/catalog \
        --embedding-model models_store/embeddings.pt \
        --classifier models_store/classifier.pt \
        --output models_store/recommendation_index.pkl

Expects `--data-dir` laid out as `<data-dir>/<category>/<image>.jpg`
(same layout as training). Runs the classifier to fill in each item's
`category` metadata field and the embedder to produce the vectors the
`RecommendationEngine` searches over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from wearnex.config import CLOTHING_CATEGORIES, MODELS_DIR
from wearnex.data.io import list_image_paths, load_image
from wearnex.data.preprocess import ClothingPreprocessor
from wearnex.models.classifier import ClothingClassifier
from wearnex.models.embeddings import EmbeddingExtractor
from wearnex.recommendation.engine import RecommendationEngine
from wearnex.training.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "embeddings.pt")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "classifier.pt")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "recommendation_index.pkl")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    preprocessor = ClothingPreprocessor(mode="eval")
    embedder = EmbeddingExtractor.load(args.embedding_model, map_location=str(device)).to(device)
    classifier = ClothingClassifier.load(
        args.classifier, map_location=str(device), num_categories=len(CLOTHING_CATEGORIES)
    ).to(device)

    paths = list_image_paths(args.data_dir, recursive=True)
    if not paths:
        raise SystemExit(f"No images found under {args.data_dir}")

    records: list[dict] = []
    embeddings: list = []

    for i in tqdm(range(0, len(paths), args.batch_size), desc="embedding catalog"):
        batch_paths = paths[i:i + args.batch_size]
        images = [load_image(p) for p in batch_paths]
        batch = torch.stack([preprocessor(img) for img in images]).to(device)

        batch_embeddings = embedder.encode(batch)
        batch_predictions = classifier.predict(batch, top_k=1)

        for path, embedding, prediction in zip(batch_paths, batch_embeddings, batch_predictions):
            records.append({
                "item_id": path.stem,
                "category": prediction["category"],
                "color": prediction["color"],
                "image_path": str(path),
            })
            embeddings.append(embedding)

    metadata = pd.DataFrame.from_records(records)
    engine = RecommendationEngine()
    engine.build_index(embeddings=np.stack(embeddings), metadata=metadata)
    engine.save(args.output)
    print(f"Indexed {len(metadata)} items -> {args.output}")


if __name__ == "__main__":
    main()
