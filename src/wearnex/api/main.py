"""FastAPI service exposing classification + recommendations to the web app.

Run locally with:
    uvicorn wearnex.api.main:app --reload

The inference pipeline (and its model weights) is loaded lazily on
first request rather than at import time, so the API can still start
— and report a clear 503 — before any models have been trained.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from wearnex.api.schemas import AnalyzeResponse, RecommendationsResponse
from wearnex.config import AI_NAME, APP_NAME
from wearnex.data.io import ImageLoadError, load_image_from_bytes
from wearnex.inference.predict import InferencePipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_NAME,
    description=f"Clothing classification and style recommendations, powered by {AI_NAME}.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> InferencePipeline:
    """Load model weights + recommendation index once, on first use."""
    try:
        return InferencePipeline()
    except FileNotFoundError as exc:
        logger.error("Model artifacts not found: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Models are not yet trained/available. Run the training scripts first.",
        ) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Classify an uploaded garment photo and return visually similar catalog items."""
    data = await file.read()
    try:
        image = load_image_from_bytes(data)
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pipeline = get_pipeline()
    result = pipeline.analyze(image)
    return AnalyzeResponse(**result)


@app.post("/recommend/similar", response_model=RecommendationsResponse)
async def recommend_similar(file: UploadFile = File(...), k: int = 10) -> RecommendationsResponse:
    """Return catalog items visually similar to an uploaded photo."""
    data = await file.read()
    try:
        image = load_image_from_bytes(data)
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pipeline = get_pipeline()
    items = pipeline.recommend_similar(image, k=k)
    return RecommendationsResponse(items=items.to_dict(orient="records"))


@app.get("/recommend/complementary/{item_id}", response_model=RecommendationsResponse)
def recommend_complementary(item_id: str, k: int = 10) -> RecommendationsResponse:
    """Return catalog items that pair well with a known catalog item (by id)."""
    pipeline = get_pipeline()
    try:
        items = pipeline.recommend_complementary(item_id, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecommendationsResponse(items=items.to_dict(orient="records"))
