"""Central configuration: paths, categories, and model hyperparameters.

Every other module pulls constants from here so that data layout, label
sets, and model shapes stay in one place as the project grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "WearNex"
AI_NAME = "JeanClaude"  # the assistant persona surfaced to users in the web app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models_store"

IMAGE_SIZE = (224, 224)  # (H, W), matches ImageNet-pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Garment categories the classifier predicts. Replace/extend to match
# whatever labeled dataset is used for training (e.g. DeepFashion2).
CLOTHING_CATEGORIES = [
    "t_shirt",
    "shirt",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "dress",
    "skirt",
    "shorts",
    "pants",
    "jeans",
    "shoes",
    "bag",
    "hat",
    "accessory",
]

# Secondary attributes usable for finer-grained recommendation matching.
COLOR_ATTRIBUTES = [
    "black", "white", "gray", "beige", "brown",
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "multicolor",
]

# Style/vibe tags usable as a secondary item attribute (distinct from
# OUTFIT_SLOTS below, which is about outfit *structure*, not aesthetics).
OUTFIT_STYLES = [
    "casual", "formal", "sporty", "business",
    "sleepwear", "loungewear", "swimwear", "snowwear",
]

OCCASION_CATEGORIES = [
    "work", "wedding", "vacation", "gym", "date",
    "night_out", "beach", "party", "hiking", "skiing",
    "running", "cycling", "travel", "concert", "festival",
    "holiday", "casual_outing",
]

SEASON_CATEGORIES = ["spring", "summer", "fall", "winter"]

# The structural positions an outfit assembler fills, and which
# CLOTHING_CATEGORIES entry belongs in each. "dress" is its own slot
# since a dress fills both "top" and "bottom" at once.
OUTFIT_SLOTS = ["top", "bottom", "dress", "outerwear", "footwear", "bag", "headwear", "accessory"]

CATEGORY_TO_SLOT: dict[str, str] = {
    "t_shirt": "top",
    "shirt": "top",
    "sweater": "top",
    "hoodie": "top",
    "jacket": "outerwear",
    "coat": "outerwear",
    "dress": "dress",
    "skirt": "bottom",
    "shorts": "bottom",
    "pants": "bottom",
    "jeans": "bottom",
    "shoes": "footwear",
    "bag": "bag",
    "hat": "headwear",
    "accessory": "accessory",
}


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_split: float = 0.15
    seed: int = 42
    categories: list[str] = field(default_factory=lambda: list(CLOTHING_CATEGORIES))


DEFAULT_TRAINING_CONFIG = TrainingConfig()
