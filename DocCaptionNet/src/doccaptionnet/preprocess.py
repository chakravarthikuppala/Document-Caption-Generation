# DocCaptionNet
from __future__ import annotations

from PIL import Image


def prepare_image(image: Image.Image, max_side: int = 384) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)
