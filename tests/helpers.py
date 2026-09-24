from pathlib import Path

from PIL import Image, ImageDraw


def solid_design(path: Path, w: int, h: int, color=(230, 60, 40, 255)) -> Path:
    """An opaque ellipse on a transparent background, like a simple print file."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((0, 0, w - 1, h // 3), fill=color)
    ImageDraw.Draw(im).ellipse((0, h // 3, w - 1, h - 1), fill=(40, 90, 200, 255))
    im.save(path)
    return path
