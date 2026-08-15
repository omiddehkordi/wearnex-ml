import io

import pytest
from PIL import Image

from stylegpt.data.io import ImageLoadError, load_image_from_bytes


def _png_bytes(size=(50, 80), color=(200, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_from_bytes_returns_rgb():
    image = load_image_from_bytes(_png_bytes())
    assert image.mode == "RGB"
    assert image.size == (50, 80)


def test_load_image_from_bytes_rejects_garbage():
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"not an image")
