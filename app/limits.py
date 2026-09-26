"""Decode limits for untrusted input (T-9-5, B-19): a format allowlist by magic bytes, a check
that the key's declared type (its extension) matches them, and a pixel cap applied before any
pixels are decoded (raster headers, SVG size, and PDF page size x DPI)."""

from pathlib import Path

from app.config import settings

# Formats we decode. SVG and PDF are vector art (T-9-1); everything else is refused.
ALLOWED_FORMATS = ("png", "jpeg", "webp", "tiff", "svg", "pdf")

# Key extension -> the format its magic bytes must show.
EXT_FORMATS = {
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "webp": "webp",
    "tif": "tiff",
    "tiff": "tiff",
    "svg": "svg",
    "pdf": "pdf",
}


def sniff_format(path: Path) -> str | None:
    """The format by magic bytes, or None when it isn't one we accept."""
    with open(path, "rb") as f:
        head = f.read(512)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head.startswith(b"%PDF-"):
        return "pdf"
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in head):
        return "svg"
    return None


def unsupported_message() -> str:
    return "unsupported image format: only PNG, JPEG, WebP, TIFF, SVG and PDF are accepted"


def declared_format(key: str) -> str | None:
    """The format a storage key declares by its extension (`…/<uuid>.png`), or None."""
    name = key.rsplit("/", 1)[-1]
    if "." not in name:
        return None
    return EXT_FORMATS.get(name.rsplit(".", 1)[-1].lower(), "unknown")


def declared_type_error(key: str, path: Path) -> str | None:
    """Why the file at `path` doesn't match the type its key declares, or None."""
    declared = declared_format(key)
    if declared is None:
        return None
    actual = sniff_format(path)
    if actual is None or declared == "unknown":
        return unsupported_message()
    if actual != declared:
        return f"file content is {actual.upper()} but the key declares {declared.upper()}"
    return None


def pixel_budget_error(width_px: float, height_px: float, what: str = "image") -> str | None:
    pixels = width_px * height_px
    cap = settings.imaging_max_decode_pixels
    if pixels > cap:
        return (
            f"{what} is {round(width_px)} x {round(height_px)} px ({pixels / 1e6:.0f}M pixels), "
            f"over the {cap / 1e6:.0f}M pixel limit"
        )
    return None


def pdf_page_budget_error(width_pt: float, height_pt: float, dpi: float) -> str | None:
    """PDF input is rasterized by pdfium, not libvips, so the decode cap is checked here from
    the page size before rendering (T-9-5 AC7)."""
    w_px, h_px = width_pt / 72.0 * dpi, height_pt / 72.0 * dpi
    return pixel_budget_error(
        w_px,
        h_px,
        f"pdf page ({width_pt / 72.0:.2f} x {height_pt / 72.0:.2f} in at {dpi:g} DPI)",
    )
