from PIL import Image

from wearnex.data.preprocess import ClothingPreprocessor, resize_with_padding


def test_resize_with_padding_preserves_target_size():
    image = Image.new("RGB", (300, 100), color=(10, 20, 30))
    result = resize_with_padding(image, size=(224, 224))
    assert result.size == (224, 224)


def test_resize_with_padding_keeps_aspect_ratio_via_padding():
    # A wide image should end up letterboxed (padding top/bottom), not stretched.
    image = Image.new("RGB", (400, 100), color=(255, 0, 0))
    result = resize_with_padding(image, size=(224, 224))
    # Padding color (white) should appear near the top edge.
    assert result.getpixel((112, 0)) == (255, 255, 255)


def test_preprocessor_output_shape_and_type():
    preprocessor = ClothingPreprocessor(mode="eval", crop_subject=False)
    image = Image.new("RGB", (180, 260), color=(30, 30, 30))
    tensor = preprocessor(image)
    assert tensor.shape == (3, 224, 224)
