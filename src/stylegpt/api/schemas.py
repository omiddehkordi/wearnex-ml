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
