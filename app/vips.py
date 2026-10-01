"""pyvips helpers shared by the endpoints."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyvips
from PIL import Image

from app.config import settings
from app.limits import pixel_budget_error, sniff_format, unsupported_message

MM_PER_IN = 25.4

# Raster loaders by sniffed format (T-9-5): never let libvips pick a loader from the content,
# and block the loaders libvips marks untrusted (magick, matload, ...) as a second line.
_RASTER_LOADERS = {
    "png": pyvips.Image.pngload,
    "jpeg": pyvips.Image.jpegload,
    "webp": pyvips.Image.webpload,
    "tiff": pyvips.Image.tiffload,
}
pyvips.block_untrusted_set(True)
# librsvg is marked untrusted, but SVG art is a feature (T-9-1): it's only ever reached through
# `load()` after sniffing, and its size is capped before rendering like any other decode.
pyvips.operation_block_set("VipsForeignLoadSvg", False)


class ImageError(Exception):
    """A user-facing image problem (unreadable input, bad geometry). Mapped to HTTP 422."""


@contextmanager
def workdir():
    with tempfile.TemporaryDirectory(prefix="invai-", dir=settings.tmp_dir) as d:
        yield Path(d)


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
    kind = sniff_format(path)
    if kind is None:
        raise ImageError(unsupported_message())
    try:
        if kind == "pdf":
            if target_dpi is None:
                raise ImageError("pdf input requires target_dpi")
            from app.pdf_input import rasterize_pdf

            img = rasterize_pdf(path, target_dpi)
        elif kind == "svg":
            scale = target_dpi / 72.0 if target_dpi is not None else 1.0
            img = pyvips.Image.svgload(str(path), scale=scale, access=access)
        else:
            img = _RASTER_LOADERS[kind](str(path), access=access)
        # The header parse (lazy, no pixels yet) makes unreadable files fail here, and the pixel
        # cap stops a decompression bomb before a single pixel is decoded.
        if err := pixel_budget_error(img.width, img.height):
            raise ImageError(err)
        return img
    except pyvips.Error as e:
        raise ImageError(f"unreadable image: {str(e).splitlines()[0]}") from e


def to_srgba(img: pyvips.Image) -> pyvips.Image:
    """Any input -> 4-band 8-bit sRGB with alpha.

    Color-managed through the file's embedded ICC profile when it carries one (B-103): a plain
    `colourspace("srgb")` only converts the color *model* (e.g. CMYK -> RGB by formula) and
    completely ignores an embedded profile, which would silently shift colors for a wide-gamut
    RGB file or a profiled CMYK print file -- the pixels would be used as-is and just relabeled
    sRGB. `icc_transform` is only called when the file actually carries a profile
    (`icc-profile-data` is present): measured live, it costs ~2ms on a 1.3MP design and ~20ms on
    a 15MP one even when there's nothing to transform, which is a real, budget-breaking slowdown
    across a full sheet of designs if paid unconditionally. CMYK input without a profile (the
    common case) still converts correctly, never silently rejected, through the same
    `colourspace()` conversion below that every input already went through -- verified live that
    it produces the same numbers `icc_transform`'s own no-profile fallback would have. A
    malformed profile libvips can't parse at all falls back to that naive conversion too, rather
    than failing the whole request.
    """
    if img.get_typeof("icc-profile-data") != 0:
        try:
            img = img.icc_transform("srgb", embedded=True, intent="perceptual")
        except pyvips.Error:
            pass
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
