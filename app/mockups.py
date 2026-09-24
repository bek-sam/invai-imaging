"""Flat t-shirt mockups: a vector-drawn tee, tinted by the blank's color, with the design on the chest.

The silhouette and its shading maps are drawn once per size (cached) from bezier paths on a 1000-unit
grid, supersampled 2x for smooth edges. Tinting multiplies a shadow map and screens a highlight map
over the flat color, so both white and black blanks show folds.
"""

from functools import lru_cache
from pathlib import Path

import pyvips
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.fonts import parse_hex
from app.vips import load, to_pil

BACKGROUND = (243, 244, 246, 255)
SS = 2  # supersampling factor


def _bezier(p0, p1, p2, p3, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        pts.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return pts


def _mirror(pts):
    return [(1000 - x, y) for x, y in reversed(pts)]


def _half_outline(neck_dip: float):
    """Right half of the tee, from the neckline center down to the hem center."""
    pts = []
    pts += _bezier((500, neck_dip), (548, neck_dip), (580, 150), (596, 112))  # neckline
    pts += _bezier((596, 112), (640, 124), (690, 136), (728, 150))  # shoulder
    pts += _bezier((728, 150), (786, 170), (838, 250), (872, 360))  # sleeve top
    pts += _bezier((872, 360), (850, 376), (812, 392), (780, 400))  # cuff
    pts += _bezier((780, 400), (760, 384), (740, 366), (722, 348))  # under sleeve
    pts += _bezier((722, 348), (704, 520), (708, 720), (722, 902))  # side seam
    pts += _bezier((722, 902), (650, 912), (560, 914), (500, 914))  # hem
    return pts


def _outline(neck_dip: float):
    right = _half_outline(neck_dip)
    return right + _mirror(right)[1:]


@lru_cache(maxsize=8)
def shirt_maps(size: int, placement: str):
    """Returns (mask, shadow, highlight) as 'L' images of size x size."""
    s = size * SS
    k = s / 1000

    def sc(pts):
        return [(x * k, y * k) for x, y in pts]

    front = placement == "front"
    neck_dip = 196 if front else 142
    outline = sc(_outline(neck_dip))

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).polygon(outline, fill=255)

    shadow = Image.new("L", (s, s), 255)
    sd = ImageDraw.Draw(shadow)
    highlight = Image.new("L", (s, s), 0)
    hd = ImageDraw.Draw(highlight)

    # Inside of the collar (the back of the shirt seen through the neck opening).
    back_neck = _bezier((404, 112), (440, 140), (560, 140), (596, 112))
    front_neck = _bezier((596, 112), (580, 150), (548, neck_dip), (500, neck_dip))
    front_neck += _mirror(front_neck)[1:]
    if front:
        sd.polygon(sc(back_neck + list(reversed(front_neck))), fill=150)
    # Rib collar band along the visible neckline.
    band = front_neck if front else back_neck
    sd.line(sc(band), fill=205, width=round(22 * k), joint="curve")
    hd.line(sc([(x, y - 9) for x, y in band]), fill=40, width=round(5 * k), joint="curve")

    # Sleeve seams and a stitched hem.
    for side in (1, -1):

        def mx(x, side=side):
            return x if side == 1 else 1000 - x

        seam = _bezier((mx(728), 150), (mx(716), 220), (mx(716), 300), (mx(722), 348))
        sd.line(sc(seam), fill=190, width=round(5 * k), joint="curve")
        cuff = _bezier((mx(858), 330), (mx(836), 346), (mx(802), 362), (mx(770), 372))
        sd.line(sc(cuff), fill=215, width=round(4 * k), joint="curve")
        # Soft fold from the armpit toward the chest.
        sd.polygon(
            sc([(mx(722), 350), (mx(690), 372), (mx(640), 420), (mx(660), 440), (mx(716), 380)]),
            fill=228,
        )
        # Side darkening.
        sd.polygon(sc([(mx(722), 348), (mx(722), 902), (mx(684), 906), (mx(680), 360)]), fill=232)
    hem = _bezier((282, 872), (380, 884), (620, 884), (718, 872))
    sd.line(sc(hem), fill=215, width=round(4 * k), joint="curve")
    # Broad soft highlight across the chest and a gentle waist fold.
    hd.ellipse(sc([(300, 200), (700, 640)]), fill=30)

    shadow = shadow.filter(ImageFilter.GaussianBlur(9 * k))
    highlight = highlight.filter(ImageFilter.GaussianBlur(70 * k))
    # Crisp seams on top of the soft shading.
    seams = Image.new("L", (s, s), 255)
    ImageDraw.Draw(seams).line(sc(band), fill=225, width=max(1, round(2 * k)))
    shadow = ImageChops.multiply(shadow, seams.filter(ImageFilter.GaussianBlur(0.8 * k)))

    mask = mask.filter(ImageFilter.GaussianBlur(0.6 * k))
    return (
        mask.resize((size, size), Image.Resampling.LANCZOS),
        shadow.resize((size, size), Image.Resampling.LANCZOS),
        highlight.resize((size, size), Image.Resampling.LANCZOS),
    )


def chest_box(size: int, placement: str) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the print area in px."""
    top = 0.27 if placement == "front" else 0.22
    w, h = 0.30 * size, 0.40 * size
    return round((size - w) / 2), round(top * size), round(w), round(h)


def render_mockup(design: Path, color_hex: str, placement: str, size: int) -> Image.Image:
    mask, shadow, highlight = shirt_maps(size, placement)
    color = parse_hex(color_hex)
    luma = (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255

    flat = Image.new("RGB", (size, size), color[:3])
    shirt = ImageChops.multiply(flat, Image.merge("RGB", (shadow,) * 3))
    strength = 0.3 + 0.7 * (1 - luma)
    hl = highlight.point(lambda v: min(255, round(v * strength)))
    shirt = ImageChops.screen(shirt, Image.merge("RGB", (hl,) * 3)).convert("RGBA")

    # Print the design: fit into the chest box, anchored to the top center.
    bx, by, bw, bh = chest_box(size, placement)
    load(design)  # validates the file with a clear error
    art = to_pil(pyvips.Image.thumbnail(str(design), bw, height=bh, size="both"))
    aw, ah = art.size
    ax, ay = bx + (bw - aw) // 2, by
    # Let the fabric's folds show through the ink a little.
    fold = shadow.crop((ax, ay, ax + aw, ay + ah)).point(lambda v: 255 - (255 - v) // 2)
    r, g, b, a = art.split()
    rgb = ImageChops.multiply(Image.merge("RGB", (r, g, b)), Image.merge("RGB", (fold,) * 3))
    art = Image.merge("RGBA", (*rgb.split(), a))
    shirt.alpha_composite(art, (ax, ay))

    out = Image.new("RGBA", (size, size), BACKGROUND)
    drop = Image.new("RGBA", (size, size), (15, 23, 42, 0))
    drop.putalpha(mask.point(lambda v: v * 38 // 255))
    drop = drop.filter(ImageFilter.GaussianBlur(size / 90))
    out.alpha_composite(drop, (0, round(size / 120)))
    shirt.putalpha(mask)
    out.alpha_composite(shirt)
    return out.convert("RGB")
