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


def _sniff(path: Path) -> str | None:
    """Detect PDF/SVG by content, not extension: callers download to extensionless temp paths."""
    with open(path, "rb") as f:
        head = f.read(256)
    if head.startswith(b"%PDF-"):
        return "pdf"
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith((b"<?xml", b"<svg")):
        return "svg"
    return None


def load(path: Path, access: str = "random", target_dpi: float | None = None) -> pyvips.Image:
    """Load an image. `target_dpi` is additive and optional: pass it to rasterize vector input
    (SVG, PDF) at that DPI instead of at their default/native size; omitting it keeps every
    existing caller's behaviour unchanged.

    SVG uses `scale=target_dpi/72`, not `dpi=target_dpi`: for an SVG with absolute physical
    units (e.g. `width="10in"`), passing `dpi=` to `svgload` double-applies the DPI (checked
    live: a 10in-wide SVG at `dpi=300` comes out 12,500px wide, not 3,000 — librsvg applies the
    factor once converting the physical size to user units, then again as an overall render
    scale). `scale=` only does the second one, which is the one we want.

    PDF has no libvips loader at all (see `app/pdf_input.py`), so it's routed there instead.
    """
    kind = _sniff(path)
    try:
        if kind == "pdf":
            if target_dpi is None:
                raise ImageError("pdf input requires target_dpi")
            from app.pdf_input import rasterize_pdf

            img = rasterize_pdf(path, target_dpi)
        elif kind == "svg" and target_dpi is not None:
            img = pyvips.Image.svgload(str(path), scale=target_dpi / 72.0, access=access)
        else:
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
