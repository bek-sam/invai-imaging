"""Personalization: a template of named text and photo slots + buyer values -> transparent PNG
at print DPI.

Text wraps within its slot box (bounded by `max_lines` when set) and shrinks to fit. The shrink
floor is the larger of 60% of `font_size_pt` (`MIN_SCALE`) and `min_font_size_pt`'s share of it,
so a slot never shrinks past whichever floor is more generous. Text that still doesn't fit at the
floor is wrapped/clipped to the box and flagged `overflow`. A photo slot is fit (letterboxed) or
filled (cropped to cover) into its box; a photo whose native resolution would print below the
150 DPI floor at the slot's physical size is flagged `low_res_photo`.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pyvips
from PIL import Image, ImageDraw, ImageFont

from app.fonts import font, missing_chars, parse_hex
from app.qa import DPI_ERROR, effective_dpi
from app.vips import ImageError, fit_box, from_pil, load, save_png, to_srgba

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
    min_font_size_pt: float | None = None
    max_lines: int | None = None
    stroke_width_pt: float = 0
    stroke_color: str | None = None
    fit: str = "fit"
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
    missing = missing_chars(slot.font_family, text)
    if missing:
        flags.append(
            Flag(
                slot.name,
                "missing_glyphs",
                f"{slot.font_family} has no glyph for: {', '.join(missing[:5])}",
            )
        )
    return flags


def _wrap(f: ImageFont.FreeTypeFont, text: str, box_w: int, stroke: int) -> list[str]:
    """Greedy word-wrap of `text` into lines no wider than `box_w` (a single word wider than the
    box gets its own, overflowing line rather than being split mid-word)."""
    words = text.split()
    if not words:
        return [text]
    lines = [words[0]]
    for w in words[1:]:
        trial = f"{lines[-1]} {w}"
        if f.getlength(trial) + stroke * 2 <= box_w:
            lines[-1] = trial
        else:
            lines.append(w)
    return lines


def fit_lines(
    slot: Slot, text: str, dpi: int, box_w: int, box_h: int
) -> tuple[list[str], ImageFont.FreeTypeFont, bool]:
    """Wrap `text` to at most `slot.max_lines` lines and find the largest scale in
    [floor_scale, 1] where it fits the box. `floor_scale` is the larger of MIN_SCALE and
    `min_font_size_pt`'s share of `font_size_pt` (see module docstring). Returns
    (lines, font, fits)."""
    size_px = slot.font_size_pt / 72 * dpi
    stroke = round(slot.stroke_width_pt / 72 * dpi)
    floor_scale = MIN_SCALE
    if slot.min_font_size_pt:
        floor_scale = max(MIN_SCALE, min(1.0, slot.min_font_size_pt / slot.font_size_pt))
    max_lines = slot.max_lines
    scale = 1.0
    while True:
        f = font(slot.font_family, max(1, round(size_px * scale)))
        lines = _wrap(f, text, box_w, stroke)
        ascent, descent = f.getmetrics()
        line_h = ascent + descent + stroke * 2
        widest = max((f.getlength(ln) + stroke * 2 for ln in lines), default=0)
        fits_lines = max_lines is None or len(lines) <= max_lines
        if fits_lines and widest <= box_w and line_h * len(lines) <= box_h:
            return lines, f, True
        if scale - SCALE_STEP < floor_scale - 1e-9:
            f = font(slot.font_family, max(1, round(size_px * floor_scale)))
            lines = _wrap(f, text, box_w, stroke)
            if max_lines is not None:
                lines = lines[:max_lines]
            return lines, f, False
        scale -= SCALE_STEP


def render_slot(slot: Slot, text: str, dpi: int) -> tuple[Image.Image, bool]:
    w, h = max(1, round(slot.w_in * dpi)), max(1, round(slot.h_in * dpi))
    stroke = round(slot.stroke_width_pt / 72 * dpi)
    lines, f, fits = fit_lines(slot, text, dpi, w, h)
    ascent, descent = f.getmetrics()
    line_h = ascent + descent + stroke * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    y0 = (h - line_h * len(lines)) / 2
    fill = parse_hex(slot.color)
    stroke_fill = parse_hex(slot.stroke_color) if slot.stroke_color else fill
    for i, line in enumerate(lines):
        left, _, right, _ = f.getbbox(line, stroke_width=stroke)
        tw = right - left
        if slot.align == "left":
            x = -left
        elif slot.align == "right":
            x = w - tw - left
        else:
            x = (w - tw) / 2 - left
        draw.text(
            (x, y0 + i * line_h),
            line,
            font=f,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=stroke_fill,
        )
    return im, fits


def render_photo_slot(slot: Slot, photo_path: Path, dpi: int) -> tuple[pyvips.Image, Flag | None]:
    """Fit (letterboxed, whole photo visible) or fill (cropped to cover) a buyer photo into its
    slot box. Flags `low_res_photo` when the photo's native resolution would print below the
    150 DPI floor at the slot's physical size."""
    w, h = max(1, round(slot.w_in * dpi)), max(1, round(slot.h_in * dpi))
    photo = to_srgba(load(photo_path))
    flag = None
    dpi_at_size = effective_dpi(photo.width, photo.height, slot.w_in, slot.h_in)
    if dpi_at_size is not None and dpi_at_size < DPI_ERROR:
        flag = Flag(
            slot.name,
            "low_res_photo",
            f"{dpi_at_size:.0f} DPI at {slot.w_in:g}x{slot.h_in:g}in; at least {DPI_ERROR} "
            "is needed",
        )
    if slot.fit == "fill":
        scale = max(w / photo.width, h / photo.height)
        rw = max(1, round(photo.width * scale))
        rh = max(1, round(photo.height * scale))
        resized = photo.resize(rw / photo.width, vscale=rh / photo.height)
        left, top = round((rw - w) / 2), round((rh - h) / 2)
        img = resized.crop(left, top, w, h)
    else:
        rw, rh = fit_box(photo.width, photo.height, w, h)
        resized = photo.resize(rw / photo.width, vscale=rh / photo.height)
        canvas = pyvips.Image.black(w, h, bands=4).copy(interpretation="srgb")
        img = canvas.composite2(resized, "over", x=round((w - rw) / 2), y=round((h - rh) / 2))
    return img, flag


def render_personalization(
    width_in: float,
    height_in: float,
    slots: list[Slot],
    values: dict[str, str],
    dpi: int,
    dest: Path,
    background: Path | None = None,
    photos: dict[str, Path] | None = None,
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

    photos = photos or {}
    result = RenderResult(W, H)
    for slot in slots:
        x, y = round(slot.x_in * dpi), round(slot.y_in * dpi)
        if slot.kind == "photo":
            photo_path = photos.get(slot.name)
            if photo_path is None:
                result.flags.append(Flag(slot.name, "empty", "no photo given"))
                continue
            img, flag = render_photo_slot(slot, photo_path, dpi)
            if flag:
                result.flags.append(flag)
            canvas = canvas.composite2(img, "over", x=x, y=y)
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
            if slot.stroke_color:
                parse_hex(slot.stroke_color)
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
        canvas = canvas.composite2(from_pil(im), "over", x=x, y=y)

    canvas = to_srgba(canvas)
    save_png(canvas, dest, dpi=dpi)
    return result
