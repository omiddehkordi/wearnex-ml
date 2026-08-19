"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class CategoryPrediction(BaseModel):
    label: str
    confidence: float


class ClassificationResult(BaseModel):
    category: str
    category_confidence: float
    top_categories: list[CategoryPrediction]
    color: str
    color_confidence: float


class RecommendedItem(BaseModel):
    item_id: str
    category: str
    image_path: str
    distance: float


class AnalyzeResponse(BaseModel):
    classification: ClassificationResult
    similar_items: list[RecommendedItem]


class RecommendationsResponse(BaseModel):
    items: list[RecommendedItem]


class OutfitRequest(BaseModel):
    """Assemble an outfit from a set of candidate items (e.g. a user's closet)."""

    item_ids: list[str]
    occasion: str | None = None
    season: str | None = None
    num_outfits: int = 1


class Outfit(BaseModel):
    slots: dict[str, RecommendedItem]
    score: float


class OutfitResponse(BaseModel):
    occasion: str | None
    season: str | None
    outfits: list[Outfit]
