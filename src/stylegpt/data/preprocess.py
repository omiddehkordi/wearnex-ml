"""Image preprocessing: crop-to-subject, resize/pad, and tensor conversion.

This is the shared pipeline used by training, batch inference, and the
future web API, so that a model always sees images prepared exactly the
way it was trained on.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from stylegpt.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def crop_to_subject(image: Image.Image, background_margin: int = 10) -> Image.Image:
    """Crop out flat, near-uniform borders around a garment photo.

    Many product-style clothing photos are shot on a plain white/gray
    backdrop with a lot of empty margin. This uses a simple threshold on
    the difference from the image's corner (background) color to find
    the subject's bounding box, and crops to it. If no clear subject
    edge is found (e.g. a busy/lifestyle photo), the original image is
    returned unchanged rather than risking a bad crop.
    """
    arr = np.array(image.convert("L"))
    corner_samples = np.concatenate(
        [arr[:5, :5].ravel(), arr[-5:, -5:].ravel(), arr[:5, -5:].ravel(), arr[-5:, :5].ravel()]
    )
    background_level = int(np.median(corner_samples))

    diff = np.abs(arr.astype(np.int16) - background_level)
    mask = (diff > 15).astype(np.uint8)

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return image

    top, bottom = max(ys.min() - background_margin, 0), min(ys.max() + background_margin, arr.shape[0])
    left, right = max(xs.min() - background_margin, 0), min(xs.max() + background_margin, arr.shape[1])

    # Guard against degenerate crops (e.g. a single stray noisy pixel).
    if (bottom - top) < 0.1 * arr.shape[0] or (right - left) < 0.1 * arr.shape[1]:
        return image

    return image.crop((left, top, right, bottom))


def resize_with_padding(image: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """Resize preserving aspect ratio, padding the rest with white.

    Plain resizing distorts garment proportions (a shoe can end up
    looking stretched into a square), which hurts both classification
    and any silhouette-sensitive similarity matching downstream.
    """
    target_h, target_w = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def denoise(image: Image.Image) -> Image.Image:
    """Light denoising for phone-camera photos (no-op for clean/studio shots)."""
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    arr = cv2.fastNlMeansDenoisingColored(arr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


class ClothingPreprocessor:
    """End-to-end preprocessing pipeline: PIL image in, model-ready tensor out.

    `mode="train"` adds label-preserving augmentation (flips, color/
    rotation jitter); `mode="eval"` is deterministic, used for
    validation, inference, and building the recommendation index so
    embeddings stay stable across runs.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_SIZE,
        mode: str = "eval",
        crop_subject: bool = True,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.image_size = image_size
        self.mode = mode
        self.crop_subject = crop_subject
        self._tensor_transform = self._build_tensor_transform()

    def _build_tensor_transform(self) -> transforms.Compose:
        ops: list = []
        if self.mode == "train":
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.RandomRotation(degrees=8, fill=255),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        return transforms.Compose(ops)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply the non-tensor steps, returning a PIL image (useful for previews/debugging)."""
        if self.crop_subject:
            image = crop_to_subject(image)
        return resize_with_padding(image, self.image_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Run the full pipeline, returning a normalized CHW tensor ready for a model."""
        image = self.preprocess_image(image)
        return self._tensor_transform(image)
