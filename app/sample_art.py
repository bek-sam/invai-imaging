"""Seed artwork: bold slogan art on a transparent background, deterministic per text.

Solid colors only (a hard drop shadow and an outline, no soft glow) so the art passes the DTF
soft-alpha check like a real print file would.
"""

import hashlib
import math

from PIL import Image, ImageDraw

from app.fonts import font, parse_hex, text_box

STYLES = ("stacked", "badge", "stars", "script")


def _mix(c, other, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c[:3], other[:3], strict=True)) + (255,)


def _luma(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _split_lines(words: list[str], max_lines: int) -> list[str]:
    if len(words) <= 1:
        return [" ".join(words)]
    n = min(max_lines, len(words))
    # Balance lines by character count.
    total = sum(len(w) for w in words) + len(words) - 1
    target = total / n
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > target * 1.15 and len(lines) < n - 1:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    lines.append(" ".join(cur))
    return lines


def _star(draw, cx, cy, r, fill):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=fill)


def _fit_lines(lines, family, box_w, box_h, stroke_ratio, line_gap=0.12):
    """Largest font size such that every line fits box_w and the stack fits box_h."""
    lo, hi = 4, max(8, box_h)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = font(family, mid)
        stroke = max(1, round(mid * stroke_ratio))
        heights = [f.getbbox("HIgjy", stroke_width=stroke) for _ in lines]
        lh = heights[0][3] - heights[0][1]
        widths = [text_box(f, ln, stroke)[2] - text_box(f, ln, stroke)[0] for ln in lines]
        total_h = lh * len(lines) + lh * line_gap * (len(lines) - 1)
        if max(widths) <= box_w and total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_sample_art(text: str, width_px: int, height_px: int, color_hex: str) -> Image.Image:
    color = parse_hex(color_hex, (30, 30, 30, 255))
    digest = hashlib.sha256(text.encode()).digest()
    style = STYLES[digest[0] % len(STYLES)]
    dark = _luma(color) < 110
    shadow = _mix(color, (255, 255, 255) if dark else (0, 0, 0), 0.55)
    outline = (255, 255, 255, 255) if dark else _mix(color, (0, 0, 0), 0.75)

    im = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    pad = round(min(width_px, height_px) * 0.06)
    words = text.split() or [" "]

    if style == "script":
        family, lines, stroke_ratio = "Pacifico", _split_lines(words, 2), 0.05
    elif style == "badge":
        family, lines, stroke_ratio = "Oswald", _split_lines(text.upper().split(), 3), 0.0
    elif style == "stars":
        family, lines, stroke_ratio = "Bebas Neue", _split_lines(text.upper().split(), 3), 0.04
    else:
        family, lines, stroke_ratio = "Inter Black", _split_lines(text.upper().split(), 3), 0.04

    box_x0, box_y0, box_x1, box_y1 = pad, pad, width_px - pad, height_px - pad
    if style == "badge":
        ring = max(4, round(min(width_px, height_px) * 0.025))
        draw.rounded_rectangle(
            (box_x0, box_y0, box_x1, box_y1),
            radius=round(min(width_px, height_px) * 0.12),
            outline=color,
            width=ring,
        )
        inner = ring * 3
        box_x0, box_y0, box_x1, box_y1 = (
            box_x0 + inner,
            box_y0 + inner,
            box_x1 - inner,
            box_y1 - inner,
        )
    if style == "stars":
        band = round((box_y1 - box_y0) * 0.16)
        r = band * 0.42
        cx = width_px / 2
        for dx in (-2.2, 0, 2.2):
            _star(draw, cx + dx * r * 1.3, box_y1 - band / 2, r, color)
        bar_h = max(3, round(band * 0.12))
        y = box_y1 - band / 2
        draw.rectangle((box_x0, y - bar_h / 2, cx - 3.6 * r, y + bar_h / 2), fill=color)
        draw.rectangle((cx + 3.6 * r, y - bar_h / 2, box_x1, y + bar_h / 2), fill=color)
        box_y1 -= band + pad // 2

    box_w, box_h = box_x1 - box_x0, box_y1 - box_y0
    size = _fit_lines(lines, family, box_w * 0.96, box_h * 0.9, stroke_ratio)
    f = font(family, size)
    stroke = round(size * stroke_ratio)
    ref = f.getbbox("HIgjy", stroke_width=stroke)
    lh = ref[3] - ref[1]
    gap = lh * 0.12
    total_h = lh * len(lines) + gap * (len(lines) - 1)
    y = box_y0 + (box_h - total_h) / 2 - ref[1]
    offset = max(2, round(size * 0.06))
    for ln in lines:
        l, _, r, _ = text_box(f, ln, stroke)
        x = box_x0 + (box_w - (r - l)) / 2 - l
        draw.text(
            (x + offset, y + offset),
            ln,
            font=f,
            fill=shadow,
            stroke_width=stroke,
            stroke_fill=shadow,
        )
        draw.text((x, y), ln, font=f, fill=color, stroke_width=stroke, stroke_fill=outline)
        y += lh + gap
    return im
