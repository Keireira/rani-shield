"""Consistent image decoding, geometry and benchmark transformations."""
from __future__ import annotations

import io

from PIL import Image, ImageOps


def load_rgb(path):
    """Decode an oriented RGB image detached from its input file."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def fit_dimensions(width, height, max_size):
    """Fit inside a square while preserving aspect ratio; never enlarge."""
    if min(width, height, max_size) <= 0:
        raise ValueError("Image dimensions and max_size must be positive.")
    ratio = min(1., max_size / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def fit_rgb(path, max_size):
    image = load_rgb(path)
    size = fit_dimensions(*image.size, max_size)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def condition_image(image, condition):
    """Apply the same post-protection transformation to every test input."""
    if condition == "png":
        return image.copy()
    if condition != "jpeg95":
        raise ValueError("Unknown input transformation.")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")
