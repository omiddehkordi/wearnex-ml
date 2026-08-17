"""PyTorch `Dataset` implementations for loading labeled clothing images.

Two construction paths are supported since labeled fashion datasets show
up in both layouts in practice:
  - `from_folder`: ImageFolder-style, one subdirectory per class.
  - `from_csv`: a metadata file with explicit image path + label columns
    (e.g. DeepFashion-style annotation exports).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from wearnex.config import CLOTHING_CATEGORIES
from wearnex.data.io import ImageLoadError, list_image_paths, load_image
from wearnex.data.preprocess import ClothingPreprocessor


class ClothingDataset(Dataset):
    """Maps (image path, label) pairs to (preprocessed tensor, label) pairs.

    Images that fail to decode are dropped at construction time (logged
    by the caller via `skipped`) rather than raising mid-training.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_to_idx: dict[str, int],
        preprocessor: ClothingPreprocessor | None = None,
    ) -> None:
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        self.preprocessor = preprocessor or ClothingPreprocessor(mode="eval")
        self.skipped: list[Path] = []
        self.samples = self._validate(samples)

    def _validate(self, samples: list[tuple[Path, int]]) -> list[tuple[Path, int]]:
        valid = []
        for path, label in samples:
            if path.exists():
                valid.append((path, label))
            else:
                self.skipped.append(path)
        return valid

    @classmethod
    def from_folder(
        cls,
        root_dir: str | Path,
        preprocessor: ClothingPreprocessor | None = None,
    ) -> "ClothingDataset":
        """Build from a directory laid out as `root_dir/<category>/<image>.jpg`."""
        root_dir = Path(root_dir)
        class_names = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        samples = [
            (path, class_to_idx[class_dir.name])
            for class_dir in root_dir.iterdir() if class_dir.is_dir()
            for path in list_image_paths(class_dir, recursive=True)
        ]
        return cls(samples, class_to_idx, preprocessor)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        image_root: str | Path,
        preprocessor: ClothingPreprocessor | None = None,
        path_column: str = "image_path",
        label_column: str = "category",
        categories: list[str] = CLOTHING_CATEGORIES,
    ) -> "ClothingDataset":
        """Build from a metadata CSV with columns `path_column`, `label_column`."""
        image_root = Path(image_root)
        df = pd.read_csv(csv_path)
        class_to_idx = {name: idx for idx, name in enumerate(categories)}

        samples = [
            (image_root / row[path_column], class_to_idx[row[label_column]])
            for _, row in df.iterrows()
            if row[label_column] in class_to_idx
        ]
        return cls(samples, class_to_idx, preprocessor)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            image = load_image(path)
        except (ImageLoadError, FileNotFoundError) as exc:
            raise RuntimeError(f"Failed to load sample at index {index}: {path}") from exc
        return self.preprocessor(image), label
