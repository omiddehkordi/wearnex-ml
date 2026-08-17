"""Image loading utilities.

These are the boundary functions between "bytes somewhere" (disk, an
HTTP upload, a URL) and a validated, correctly-oriented PIL image that
the rest of the pipeline can rely on. Every loader here funnels through
`_finalize_image` so callers get consistent behavior regardless of
source.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageLoadError(ValueError):
    """Raised when image bytes cannot be decoded into a usable image."""


def _finalize_image(image: Image.Image) -> Image.Image:
    """Normalize orientation and color mode for any loaded image.

    - Applies EXIF orientation (phone photos are often stored sideways).
    - Converts to RGB so downstream code never has to special-case
      grayscale, palette (P), or RGBA images.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_image(path: str | Path) -> Image.Image:
    """Load an image from a local file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            img.load()
            return _finalize_image(img)
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Could not decode image: {path}") from exc


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load an image from raw bytes (e.g. a web upload's request body)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return _finalize_image(img)
    except UnidentifiedImageError as exc:
        raise ImageLoadError("Could not decode image bytes") from exc


def load_image_from_url(url: str, timeout: float = 10.0) -> Image.Image:
    """Fetch and decode an image from a remote URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return load_image_from_bytes(response.content)


def list_image_paths(directory: str | Path, recursive: bool = True) -> list[Path]:
    """List all supported image files under `directory`.

    Skips files that don't match a known image extension; logs and skips
    files that exist but fail to decode, rather than aborting the whole
    listing over one bad file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)
