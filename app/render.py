"""Personalization: a template of named text slots + buyer values -> transparent PNG at print DPI.

Each value is set in its slot's font at font_size_pt; if it doesn't fit, it shrinks down to 60% of
that size, and past that it is rendered at 60% clipped to the slot and flagged `overflow`.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pyvips
from PIL import Image, ImageDraw

from app.fonts import font, parse_hex
from app.vips import ImageError, from_pil, load, save_png, to_srgba

MIN_SCALE = 0.6
SCALE_STEP = 0.02
# Latin script, digits and ordinary punctuation. Anything else (emoji, symbols, other scripts,
# control characters) is flagged for a human to look at; the bundled fonts may not have it.
_ALLOWED = re.compile(r"^[A-Za-z0-9À-ɏ\s.,'’‘\"“”&!?\-–—:;()/#+@$%*]*$")


@dataclass
class Slot:
    name: str
    x_in: float
    y_in: float
    w_in: float
    h_in: float
    font_family: str = "Inter"
    font_size_pt: float = 36
    color: str = "#000000"
    align: str = "center"
    max_chars: int | None = None
    uppercase: bool = False
    kind: str = "text"


@dataclass
class Flag:
    slot: str
    code: str
    message: str


@dataclass
class RenderResult:
    width_px: int
    height_px: int
    flags: list[Flag] = field(default_factory=list)


def check_text(slot: Slot, text: str) -> list[Flag]:
    flags = []
    if not text.strip():
        flags.append(Flag(slot.name, "empty", "no value given"))
        return flags
    if slot.max_chars and len(text) > slot.max_chars:
        flags.append(
            Flag(slot.name, "too_long", f"{len(text)} characters; the limit is {slot.max_chars}")
        )
    bad = sorted({ch for ch in text if not _ALLOWED.match(ch)})
    if bad:
        names = ", ".join(f"{ch!r} ({unicodedata.name(ch, f'U+{ord(ch):04X}')})" for ch in bad[:5])
        flags.append(Flag(slot.name, "suspicious_chars", f"unusual characters: {names}"))
    return flags


def fit_text(text: str, family: str, size_px: float, box_w: int, box_h: int):
    """Largest scale in [MIN_SCALE, 1] where the text fits the box. Returns (font, fits)."""
    scale = 1.0
    while True:
        f = font(family, max(1, round(size_px * scale)))
        left, top, right, bottom = f.getbbox(text)
        if right - left <= box_w and bottom - top <= box_h:
            return f, True
        if scale - SCALE_STEP < MIN_SCALE - 1e-9:
            return font(family, max(1, round(size_px * MIN_SCALE))), False
        scale -= SCALE_STEP


def render_slot(slot: Slot, text: str, dpi: int) -> tuple[Image.Image, bool]:
    w, h = max(1, round(slot.w_in * dpi)), max(1, round(slot.h_in * dpi))
    size_px = slot.font_size_pt / 72 * dpi
    f, fits = fit_text(text, slot.font_family, size_px, w, h)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    left, top, right, bottom = f.getbbox(text)
    tw, th = right - left, bottom - top
    if slot.align == "left":
        x = -left
    elif slot.align == "right":
        x = w - tw - left
    else:
        x = (w - tw) / 2 - left
    y = (h - th) / 2 - top
    draw.text((x, y), text, font=f, fill=parse_hex(slot.color))
    return im, fits


def render_personalization(
    width_in: float,
    height_in: float,
    slots: list[Slot],
    values: dict[str, str],
    dpi: int,
    dest: Path,
    background: Path | None = None,
) -> RenderResult:
    W, H = round(width_in * dpi), round(height_in * dpi)
    if W <= 0 or H <= 0:
        raise ImageError("template width_in and height_in must be positive")
    if background is not None:
        bg = to_srgba(load(background))
        canvas = (
            bg.resize(W / bg.width, vscale=H / bg.height)
            if (bg.width, bg.height)
            != (
                W,
                H,
            )
            else bg
        )
    else:
        canvas = pyvips.Image.black(W, H, bands=4).copy(interpretation="srgb")

    result = RenderResult(W, H)
    for slot in slots:
        if slot.kind != "text":
            continue
        text = (values.get(slot.name) or "").strip()
        if slot.uppercase:
            text = text.upper()
        flags = check_text(slot, text)
        result.flags.extend(flags)
        if not text:
            continue
        try:
            parse_hex(slot.color)
        except ValueError as e:
            raise ImageError(f"slot {slot.name}: {e}") from e
        im, fits = render_slot(slot, text, dpi)
        if not fits:
            result.flags.append(
                Flag(
                    slot.name,
                    "overflow",
                    f"text does not fit the slot even at {MIN_SCALE:.0%} of "
                    f"{slot.font_size_pt:g}pt; it is clipped",
                )
            )
        x, y = round(slot.x_in * dpi), round(slot.y_in * dpi)
        canvas = canvas.composite2(from_pil(im), "over", x=x, y=y)

    canvas = to_srgba(canvas)
    save_png(canvas, dest, dpi=dpi)
    return result
