"""pyvips helpers shared by the endpoints."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyvips
from PIL import Image

from app.config import settings

MM_PER_IN = 25.4


class ImageError(Exception):
    """A user-facing image problem (unreadable input, bad geometry). Mapped to HTTP 422."""


@contextmanager
def workdir():
    with tempfile.TemporaryDirectory(prefix="invai-", dir=settings.tmp_dir) as d:
        yield Path(d)


def load(path: Path, access: str = "random") -> pyvips.Image:
    try:
        img = pyvips.Image.new_from_file(str(path), access=access)
        # Force the header parse so unreadable files fail here, not mid-pipeline.
        _ = img.width
        return img
    except pyvips.Error as e:
        raise ImageError(f"unreadable image: {str(e).splitlines()[0]}") from e


def to_srgba(img: pyvips.Image) -> pyvips.Image:
    """Any input -> 4-band 8-bit sRGB with alpha."""
    if img.interpretation not in ("srgb", "b-w") or img.format != "uchar":
        img = img.colourspace("srgb")
    if img.bands in (1, 2):
        img = img.colourspace("srgb")
    if img.format != "uchar":
        img = img.cast("uchar")
    if not img.hasalpha():
        img = img.bandjoin(255)
    if img.bands > 4:
        img = img.extract_band(0, n=4)
    return img.copy(interpretation="srgb")


def set_dpi(img: pyvips.Image, dpi: float) -> pyvips.Image:
    return img.copy(xres=dpi / MM_PER_IN, yres=dpi / MM_PER_IN)


def from_pil(im: Image.Image) -> pyvips.Image:
    im = im.convert("RGBA")
    return pyvips.Image.new_from_memory(im.tobytes(), im.width, im.height, 4, "uchar").copy(
        interpretation="srgb"
    )


def to_pil(img: pyvips.Image) -> Image.Image:
    img = to_srgba(img)
    return Image.frombytes("RGBA", (img.width, img.height), img.write_to_memory())


def fit_box(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Largest size with the source aspect ratio that fits inside the box."""
    scale = min(box_w / src_w, box_h / src_h)
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


def save_png(img: pyvips.Image, path: Path, dpi: float | None = None, compression: int = 6):
    if dpi:
        img = set_dpi(img, dpi)
    img.pngsave(str(path), compression=compression, strip=True)
