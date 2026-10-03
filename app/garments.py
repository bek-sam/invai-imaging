"""Garment template library for listing photos (T-26-2): tee, hoodie, crewneck and tank, each
in four views (`front_flat`, `folded`, `back`, `on_model_white`).

Everything is drawn by code from bezier paths, in **inches** on an adult-L garment, so the
template's physical scale is exact: at any output size a template has one `ppi`, and a design of
W x H inches covers exactly W*ppi x H*ppi pixels. The garment coordinate system has x = 0 on the
center line (left half is the mirror of the right) and y = 0 at the shoulder/neck point, y down.

`template_maps()` draws one template at one output size (supersampled 2x, cached) and returns
the maps a compositor needs, all the output size:
- `back`, `front`: garment coverage (0..255). `back` is garment behind the figure (a hood);
  `front` is the printable garment in front of everything. The print is clipped by `front`.
- `figure`, `figure_rgb`: the drawn neutral figure for `on_model_white` (None otherwise).
- `shadow` (multiply, 255 = none), `highlight` (screen, 0 = none), `height` (128 = flat): folds,
  seams and shadows. `height`'s gradient displaces the print, so folds bend it.
- `print_area_px`, `flat_px` (a garment pixel with no shading at all, outside the print area),
  `product_bbox_px`, `ppi`.
Wave 27 (scene composite) reuses these maps and `Template.print_area`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache, partial

import pyvips
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.config import settings

GARMENTS = ("tee", "hoodie", "crewneck", "tank")
VIEWS = ("front_flat", "folded", "back", "on_model_white")
# Scene bases for AI lifestyle images (T-27-2). `lifestyle_base` is the on-model figure drawn
# smaller (LIFESTYLE_FILL), leaving room around it for the provider to draw a scene. Not in
# VIEWS: it is a base for /photo/scene-base, not a listing template.
SCENE_VIEWS = ("on_model_white", "lifestyle_base")
LIFESTYLE_FILL = 0.62
SS = 2  # supersampling
FILL = 0.90  # the product's longest side spans this share of the frame (5% margin each side)

FIGURE_RGB = (206, 205, 203)  # neutral light grey mannequin
FIGURE_EDGE = (168, 166, 163)
PANTS_RGB = (122, 126, 134)

Pt = tuple[float, float]


@dataclass(frozen=True)
class Box:
    x: float  # left, inches (garment coordinates)
    y: float  # top
    w: float
    h: float


def bz(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: int = 20) -> list[Pt]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return out


def line(a: Pt, b: Pt) -> list[Pt]:
    return [a, b]


def chain(*segments: list[Pt]) -> list[Pt]:
    pts = list(segments[0])
    for s in segments[1:]:
        pts += s[1:]
    return pts


def sym(right: list[Pt]) -> list[Pt]:
    """Closed outline from a right half drawn from the top center to the bottom center."""
    return right + [(-x, y) for x, y in reversed(right)][1:]


def across(half: list[Pt]) -> list[Pt]:
    """A full curve (left end -> center -> right end) from a half starting at the center."""
    return [(-x, y) for x, y in reversed(half)] + half[1:]


def mx(pts: list[Pt]) -> list[Pt]:
    return [(-x, y) for x, y in pts]


def side(s: int, pts: list[Pt]) -> list[Pt]:
    """The right-side points as drawn (s=1) or mirrored onto the left (s=-1)."""
    return pts if s == 1 else mx(pts)


class Canvas:
    """Pillow layers in garment inches. Coverage masks are drawn at SS x the output size (smooth
    edges); everything else at 1x: seams and shading are blurred past a pixel anyway, and the
    figure's colors sit under its 2x mask. That keeps the draw's peak memory down."""

    SOFT = ("shadow", "figure_rgb", "highlight", "soft_shadow", "soft_height")

    def __init__(self, w: int, h: int, ppi: float, x0: float, y0: float, figure: bool):
        self.w, self.h = w * SS, h * SS
        self.k = ppi * SS
        self.k1 = ppi
        self.x0, self.y0 = x0, y0
        size, size1 = (self.w, self.h), (w, h)
        self.back: Image.Image | None = None  # only a hoodie on a figure has one (created on use)
        self.front = Image.new("L", size, 0)
        self.figure = Image.new("L", size, 0) if figure else None
        self.figure_rgb = Image.new("RGB", size1, FIGURE_RGB) if figure else None
        self.shadow = Image.new("L", size1, 255)
        self.highlight = Image.new("L", size1, 0)
        self.soft_shadow = Image.new("L", size1, 255)
        self.soft_height = Image.new("L", size1, 128)
        names = ("front", "shadow", "figure", "figure_rgb", *self.SOFT)
        self.d = {n: ImageDraw.Draw(getattr(self, n)) for n in names if getattr(self, n)}

    def _k(self, layer: str | None) -> float:
        return self.k1 if layer in self.SOFT else self.k

    def p(self, pts: list[Pt], layer: str | None = None) -> list[Pt]:
        k = self._k(layer)
        return [((x - self.x0) * k, (y - self.y0) * k) for x, y in pts]

    def px(self, inches: float, layer: str | None = None) -> int:
        return max(1, round(inches * self._k(layer)))

    def poly(self, layer: str, pts: list[Pt], fill: int | tuple) -> None:
        if layer == "back" and self.back is None:
            self.back = Image.new("L", (self.w, self.h), 0)
            self.d["back"] = ImageDraw.Draw(self.back)
        self.d[layer].polygon(self.p(pts, layer), fill=fill)

    def garment(self, layer: str, pts: list[Pt], edge: list[Pt] | None = None) -> None:
        """A garment piece: coverage plus a darker rim along its silhouette (`edge`, default
        the closed outline), so a white blank still reads on a white background."""
        self.poly(layer, pts, 255)
        edge = edge if edge is not None else pts + pts[:1]
        self.stroke("shadow", edge, 200, 0.1)
        self.stroke("soft_shadow", edge, 226, 0.7)

    def stroke(self, layer: str, pts: list[Pt], fill: int | tuple, width_in: float) -> None:
        self.d[layer].line(
            self.p(pts, layer), fill=fill, width=self.px(width_in, layer), joint="curve"
        )

    def ellipse(self, layer: str, cx: float, cy: float, rx: float, ry: float, fill) -> None:
        (a, b), (c, d) = self.p([(cx - rx, cy - ry), (cx + rx, cy + ry)], layer)
        self.d[layer].ellipse((a, b, c, d), fill=fill)

    def fold(self, pts: list[Pt], depth: float = 1.0) -> None:
        """A soft fold: a shadow valley that also bends the print (height map)."""
        self.stroke("soft_shadow", pts, round(255 - 48 * depth), 0.55)
        self.stroke("soft_height", pts, round(128 - 70 * depth), 0.9)
        self.stroke("highlight", [(x, y - 0.45) for x, y in pts], round(26 * depth), 0.3)

    def seam(self, pts: list[Pt], shade: int = 196, width_in: float = 0.09) -> None:
        self.stroke("shadow", pts, shade, width_in)

    def limb(self, a: Pt, b: Pt, width_in: float) -> None:
        """A figure limb: a thick round-capped stroke with an edge line."""
        for layer, fill, extra in (("figure_rgb", FIGURE_EDGE, 0.14), ("figure", 255, 0.14)):
            self.stroke(layer, [a, b], fill, width_in + extra)
            for p in (a, b):
                r = (width_in + extra) / 2
                self.ellipse(layer, p[0], p[1], r, r, fill)
        self.stroke("figure_rgb", [a, b], FIGURE_RGB, width_in)
        for p in (a, b):
            self.ellipse("figure_rgb", p[0], p[1], width_in / 2, width_in / 2, FIGURE_RGB)

    def figure_poly(self, pts: list[Pt], rgb=FIGURE_RGB) -> None:
        self.d["figure"].polygon(self.p(pts), fill=255)
        self.d["figure_rgb"].polygon(
            self.p(pts, "figure_rgb"),
            fill=rgb,
            outline=FIGURE_EDGE,
            width=self.px(0.07, "figure_rgb"),
        )

    def figure_ellipse(self, cx: float, cy: float, rx: float, ry: float) -> None:
        self.ellipse("figure", cx, cy, rx, ry, 255)
        (a, b), (c, d) = self.p([(cx - rx, cy - ry), (cx + rx, cy + ry)], "figure_rgb")
        self.d["figure_rgb"].ellipse(
            (a, b, c, d), fill=FIGURE_RGB, outline=FIGURE_EDGE, width=self.px(0.07, "figure_rgb")
        )


# --- garment geometry (inches, adult L) ----------------------------------------------------


def tee_half(dip: float) -> list[Pt]:
    return chain(
        bz((0, dip), (2.0, dip), (3.3, dip * 0.45), (3.6, 0)),  # neckline
        bz((3.6, 0), (5.5, 0.5), (7.5, 1.1), (9.0, 1.6)),  # shoulder
        bz((9.0, 1.6), (11.0, 2.6), (13.4, 5.6), (14.6, 8.6)),  # sleeve top
        bz((14.6, 8.6), (13.9, 9.3), (13.0, 9.9), (12.2, 10.3)),  # cuff
        bz((12.2, 10.3), (11.6, 9.6), (11.0, 8.9), (10.4, 8.4)),  # under sleeve
        bz((10.4, 8.4), (10.2, 14), (10.3, 22), (10.5, 29)),  # side seam
        bz((10.5, 29), (7.5, 29.25), (3.5, 29.3), (0, 29.3)),  # hem
    )


def tee_neck(dip: float) -> list[Pt]:
    return bz((0, dip), (2.0, dip), (3.3, dip * 0.45), (3.6, 0))


def long_half(dip: float, body: float, nx: float, ny: float) -> list[Pt]:
    """Hoodie / crewneck: long set-in sleeves angled down, rib cuffs and waistband."""
    return chain(
        long_neck(dip, nx, ny),
        bz((nx, ny), (6.5, ny + 0.5), (8.5, ny + 1.0), (10.0, 2.0)),  # shoulder
        bz((10.0, 2.0), (13.5, 3.5), (17.5, 12.5), (19.5, 21.5)),  # sleeve top
        line((19.5, 21.5), (19.9, 24.9)),  # cuff, outer
        bz((19.9, 24.9), (18.8, 25.5), (17.2, 25.9), (16.2, 25.9)),  # cuff, end
        line((16.2, 25.9), (15.9, 22.6)),  # cuff, inner
        bz((15.9, 22.6), (14.8, 17.0), (13.2, 12.5), (body, 10.2)),  # under sleeve
        bz((body, 10.2), (body - 0.1, 16), (body - 0.1, 21), (body - 0.2, 25.0)),  # side
        line((body - 0.2, 25.0), (body - 0.6, 27.6)),  # waistband
        bz((body - 0.6, 27.6), (7, 27.8), (3, 27.8), (0, 27.8)),  # hem
    )


def long_neck(dip: float, nx: float, ny: float) -> list[Pt]:
    return bz((0, dip), (2.0, dip), (nx - 0.3, (dip + ny) * 0.5), (nx, ny))


def tank_neck(dip: float) -> list[Pt]:
    return bz((0, dip), (2.2, dip), (3.8, dip * 0.45), (4.0, 0))


def tank_half(dip: float) -> list[Pt]:
    return chain(
        tank_neck(dip),
        line((4.0, 0), (6.3, 0.3)),  # strap
        bz((6.3, 0.3), (6.6, 4.5), (7.8, 8.6), (9.4, 9.6)),  # armhole
        bz((9.4, 9.6), (9.3, 16), (9.4, 22), (9.6, 28)),  # side
        bz((9.6, 28), (6.5, 28.2), (3, 28.25), (0, 28.25)),  # hem
    )


@dataclass(frozen=True)
class GarmentSpec:
    front_dip: float
    back_dip: float
    neck: Callable[[float], list[Pt]]  # half neckline from the center for a dip
    outline: Callable[[float], list[Pt]]  # right half outline for a dip
    front_area: Box
    back_area: Box
    fold_half_width: float
    arms: tuple[tuple[Pt, Pt, float], ...]  # on-model arm segments (from, to, width)
    hand: Pt


HOODIE_BODY, CREW_BODY = 11.5, 11.0

SPECS: dict[str, GarmentSpec] = {
    # Adult tee: front max 12 x 16 in, top 3 in below the collar; back 14 x 16.
    "tee": GarmentSpec(
        3.6,
        0.9,
        tee_neck,
        tee_half,
        Box(-6, 6.6, 12, 16),
        Box(-7, 3.9, 14, 16),
        7.0,
        (((11.4, 5.6), (18.8, 17.0), 3.0),),
        (19.3, 17.9),
    ),
    # Hoodie: the front print sits above the kangaroo pocket (pocket top at 15.5 in).
    "hoodie": GarmentSpec(
        3.0,
        1.0,
        lambda d: long_neck(d, 4.3, 0.4),
        lambda d: long_half(d, HOODIE_BODY, 4.3, 0.4),
        Box(-6, 6.0, 12, 9),
        Box(-7, 3.5, 14, 16),
        7.6,
        (((17.2, 20.5), (18.3, 26.6), 3.0),),
        (18.5, 27.3),
    ),
    "crewneck": GarmentSpec(
        3.2,
        0.8,
        lambda d: long_neck(d, 3.9, 0.2),
        lambda d: long_half(d, CREW_BODY, 3.9, 0.2),
        Box(-6, 6.2, 12, 14),
        Box(-7, 3.8, 14, 16),
        7.4,
        (((17.2, 20.5), (18.3, 26.6), 3.0),),
        (18.5, 27.3),
    ),
    # Tank: a narrower print (10 in) between the armholes.
    "tank": GarmentSpec(
        5.5,
        2.0,
        tank_neck,
        tank_half,
        Box(-5, 7.5, 10, 13),
        Box(-5, 4.5, 10, 14),
        6.4,
        (((7.7, 1.8), (10.6, 11.5), 3.6), ((10.6, 11.5), (12.4, 21.0), 3.0)),
        (12.7, 22.6),
    ),
}


@dataclass(frozen=True)
class Template:
    garment: str
    view: str
    placement: str  # which print file the view shows: "front" or "back"
    print_area: Box
    drawn: bool = True
    crop_bottom: float | None = None  # on-model: the frame's bottom edge cuts the legs here
    fill: float = FILL  # share of the frame the product spans on its limiting side
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def spec(self) -> GarmentSpec:
        return SPECS[self.garment]

    @property
    def on_model(self) -> bool:
        return self.view in SCENE_VIEWS


def get_template(garment: str, view: str) -> Template:
    if garment not in SPECS or (view not in VIEWS and view not in SCENE_VIEWS):
        raise KeyError(f"no template for {garment}/{view}")
    spec = SPECS[garment]
    placement = "back" if view == "back" else "front"
    area = spec.back_area if placement == "back" else spec.front_area
    on_model = view in SCENE_VIEWS
    return Template(
        garment,
        view,
        placement,
        area,
        crop_bottom=34.0 if on_model else None,
        fill=LIFESTYLE_FILL if view == "lifestyle_base" else FILL,
    )


def all_templates() -> list[Template]:
    return [get_template(g, v) for g in GARMENTS for v in VIEWS]


# --- drawing --------------------------------------------------------------------------------


def _opening(spec: GarmentSpec) -> list[Pt]:
    """The neck opening seen from the front: between the back and the front necklines."""
    return across(spec.neck(spec.back_dip)) + list(reversed(across(spec.neck(spec.front_dip))))


def _draw_body(c: Canvas, t: Template, layer: str) -> None:
    spec = t.spec
    g = t.garment
    front = t.view != "back"
    dip = spec.front_dip if front else spec.back_dip
    c.garment(layer, sym(spec.outline(dip)))
    neck = across(spec.neck(dip))

    if g == "tee":
        for s in (1, -1):
            f = partial(side, s)
            c.seam(f(bz((9.0, 1.6), (9.4, 3.8), (9.8, 6.4), (10.4, 8.4))))
            c.seam(f(bz((14.2, 8.0), (13.6, 8.6), (12.7, 9.2), (11.9, 9.6))), 214, 0.07)
            c.poly(
                "soft_shadow",
                f([(10.4, 8.4), (10.0, 14), (10.2, 28.6), (8.9, 28.8), (9.0, 9.2)]),
                236,
            )
            c.fold(f(bz((10.0, 9.4), (8.8, 10.4), (7.4, 11.6), (5.8, 13.2))), 0.9)
            c.fold(f(bz((13.0, 6.2), (12.4, 6.9), (11.8, 7.6), (11.2, 8.2))), 0.6)
        c.seam(bz((-10.45, 28.35), (-4, 28.6), (4, 28.6), (10.45, 28.35)), 214, 0.07)
        c.fold(bz((-8.2, 24.9), (-3, 24.2), (3, 25.2), (8.2, 24.4)), 0.55)
        c.ellipse("highlight", 0, 12.5, 6.5, 7.5, 22)
    elif g in ("hoodie", "crewneck"):
        body = HOODIE_BODY if g == "hoodie" else CREW_BODY
        for s in (1, -1):
            f = partial(side, s)
            c.seam(f(bz((10.0, 2.0), (10.6, 6.0), (11.0, 8.0), (body, 10.2))))
            c.seam(f(line((19.5, 21.5), (15.9, 22.6))), 200, 0.09)
            for i in range(1, 6):  # cuff ribs
                a = i / 6
                c.seam(
                    f(
                        line(
                            (19.5 + 0.4 * a - a * 3.6, 21.5 + 1.1 * a + 0.0),
                            (19.9 - a * 3.7, 24.9 + a * 1.0),
                        )
                    ),
                    232,
                    0.04,
                )
            c.poly(
                "soft_shadow",
                f([(body, 10.2), (body - 0.1, 24.8), (body - 1.5, 24.8), (body - 1.4, 11.0)]),
                236,
            )
            c.fold(f(bz((17.4, 15.2), (16.6, 15.6), (15.6, 16.2), (14.6, 16.8))), 0.8)
            c.fold(f(bz((18.4, 18.6), (17.6, 19.2), (16.6, 19.6), (15.5, 20.2))), 0.6)
            c.fold(
                f(
                    bz(
                        (body - 0.6, 11.2),
                        (body - 2.0, 12.0),
                        (body - 3.6, 13.0),
                        (body - 5.0, 14.4),
                    )
                ),
                0.7,
            )
        c.seam(line((-(body - 0.2), 25.0), (body - 0.2, 25.0)), 200, 0.1)
        x = -(body - 0.6)
        while x < body - 0.6:  # waistband ribs
            c.seam(line((x, 25.1), (x * 0.97, 27.6)), 234, 0.04)
            x += 0.42
        c.ellipse("highlight", 0, 12.5, 7.0, 7.0, 22)
    else:  # tank
        for s in (1, -1):
            f = partial(side, s)
            c.stroke("shadow", f(bz((6.3, 0.3), (6.6, 4.5), (7.8, 8.6), (9.4, 9.6))), 212, 0.45)
            c.poly(
                "soft_shadow",
                f([(9.4, 9.6), (9.3, 16), (9.5, 27.6), (8.2, 27.8), (8.2, 10.4)]),
                236,
            )
            c.fold(f(bz((9.0, 10.6), (8.2, 11.6), (7.0, 12.8), (5.6, 14.0))), 0.7)
        c.seam(bz((-9.55, 27.4), (-4, 27.6), (4, 27.6), (9.55, 27.4)), 214, 0.07)
        c.fold(bz((-7.4, 23.9), (-2.5, 23.3), (2.5, 24.2), (7.4, 23.5)), 0.5)
        c.ellipse("highlight", 0, 13.5, 5.0, 7.0, 22)

    # Collar / neck band along the visible neckline.
    band = {"tee": 0.75, "hoodie": 0.3, "crewneck": 0.95, "tank": 0.45}[g]
    c.stroke("shadow", neck, 214, band)
    c.stroke("highlight", [(x, y - band * 0.35) for x, y in neck], 34, 0.06)
    if front and g != "hoodie":
        c.poly("shadow", _opening(t.spec), 150)
    if not front:
        c.seam(neck, 205, 0.06)


def _hood_curve() -> list[Pt]:
    """The hood's visible rim, shoulder to shoulder (no closing line through the body)."""
    right = chain(
        bz((0, -10.3), (3.2, -10.3), (6.4, -7.0), (6.2, -3.0)),
        bz((6.2, -3.0), (6.0, -1.0), (5.0, 0.0), (4.3, 0.4)),
    )
    return across(right)


def _hood_outer() -> list[Pt]:
    right = chain(
        bz((0, -10.3), (3.2, -10.3), (6.4, -7.0), (6.2, -3.0)),
        bz((6.2, -3.0), (6.0, -1.0), (5.0, 0.0), (4.3, 0.4)),
        line((4.3, 0.4), (0.0, 4.8)),
    )
    return sym(right)


def _hood_opening(scale: float = 1.0) -> list[Pt]:
    right = chain(
        bz((0, -7.6 * scale), (2.6, -7.6 * scale), (3.9, -4.5 * scale), (3.8, -1.5 * scale)),
        bz((3.8, -1.5 * scale), (3.7, 1.0), (2.4, 3.2), (0, 4.6)),
    )
    return sym(right)


def _draw_hoodie_front(c: Canvas, hood_layer: str, opening_shows_figure: bool) -> None:
    c.garment(hood_layer, _hood_outer(), _hood_curve())
    c.seam(line((0, -10.3), (0, -7.6)), 205, 0.08)
    opening = _hood_opening()
    if opening_shows_figure:
        c.poly("front", opening, 0)
    else:
        c.poly("shadow", opening, 145)
    c.stroke("shadow", opening + opening[:1], 222, 0.7)
    c.stroke("highlight", opening + opening[:1], 30, 0.12)
    for s in (1, -1):  # drawstrings and aglets
        c.stroke("shadow", [(1.4 * s, 3.9), (1.5 * s, 11.0)], 178, 0.25)
        c.stroke("highlight", [(1.36 * s, 3.9), (1.46 * s, 11.0)], 40, 0.06)
        c.stroke("shadow", [(1.5 * s, 10.4), (1.52 * s, 11.2)], 110, 0.3)
    # Kangaroo pocket.
    c.seam(line((-5.2, 15.5), (5.2, 15.5)), 196, 0.1)
    for s in (1, -1):
        f = partial(side, s)
        curve = bz((5.2, 15.5), (6.4, 16.2), (7.6, 19.0), (8.6, 25.0))
        c.seam(f(curve), 188, 0.11)
        c.poly("soft_shadow", f(curve + [(9.3, 25.0), (8.3, 19.0), (6.0, 15.6)]), 228)


def _draw_folded(c: Canvas, t: Template) -> None:
    spec, g = t.spec, t.garment
    fw = spec.fold_half_width
    neck = spec.neck(spec.front_dip)
    nx, ny = neck[-1]
    if g == "tank":
        top = chain(line((nx, ny), (5.8, 0.3)), bz((5.8, 0.3), (5.9, 2.8), (6.0, 4.2), (fw, 5.2)))
    else:
        top = bz((nx, ny), (nx + 1.2, ny + 0.35), (fw - 1.0, 1.0), (fw, 1.2))
    right = chain(
        neck,
        top,
        bz(top[-1], (fw + 0.05, 6), (fw + 0.05, 12), (fw, 16.0)),
        bz((fw, 16.0), (fw, 16.4), (fw - 0.3, 16.6), (fw - 0.8, 16.6)),
        line((fw - 0.8, 16.6), (0, 16.6)),
    )
    if g == "hoodie":
        hood_right = chain(
            bz((0, -4.8), (3.0, -4.8), (5.6, -3.0), (5.6, -0.6)),
            bz((5.6, -0.6), (5.6, 0.2), (5.0, 0.4), (4.3, 0.4)),
            line((4.3, 0.4), (0, 4.8)),
        )
        c.garment("front", sym(hood_right), across(hood_right[:-1]))
    c.garment("front", sym(right))
    for s in (1, -1):
        f = partial(side, s)
        c.poly(
            "soft_shadow",
            f(
                [
                    (fw - 0.55, top[-1][1]),
                    (fw + 0.1, top[-1][1]),
                    (fw + 0.1, 16.6),
                    (fw - 0.55, 16.6),
                ]
            ),
            214,
        )
        if g != "tank":
            c.seam(f(line((fw - 0.15, 4.2), (fw - 1.8, 1.15))), 204, 0.09)
    c.poly("soft_shadow", [(-fw, 15.8), (fw, 15.8), (fw, 16.8), (-fw, 16.8)], 212)
    c.fold(bz((-fw + 0.7, 10.7), (-2, 10.3), (2, 10.8), (fw - 0.7, 10.4)), 0.6)
    c.ellipse("highlight", 0, 8.0, 5.0, 5.0, 22)
    band = {"tee": 0.75, "hoodie": 0.3, "crewneck": 0.95, "tank": 0.45}[g]
    full_neck = across(neck)
    c.stroke("shadow", full_neck, 214, band)
    if g == "hoodie":
        opening = sym(
            chain(
                bz((0, -3.2), (2.2, -3.2), (3.6, -1.8), (3.6, -0.5)),
                bz((3.6, -0.5), (3.5, 1.2), (2.4, 3.2), (0, 4.6)),
            )
        )
        c.poly("shadow", opening, 145)
        c.stroke("shadow", opening + opening[:1], 222, 0.6)
        for s in (1, -1):
            c.stroke("shadow", [(1.4 * s, 3.9), (1.5 * s, 11.0)], 178, 0.25)
            c.stroke("shadow", [(1.5 * s, 10.4), (1.52 * s, 11.2)], 110, 0.3)
        c.seam(line((-5.2, 15.5), (5.2, 15.5)), 196, 0.1)
    else:
        c.poly("shadow", _opening(spec), 150)


def _draw_figure(c: Canvas, t: Template) -> None:
    spec = t.spec
    # Legs (trousers), torso, arms, neck and head; the garment is drawn over them.
    c.figure_poly(
        [
            (-8.4, 26.5),
            (8.4, 26.5),
            (8.7, 31),
            (8.5, 40),
            (0.7, 40),
            (0.4, 32.5),
            (-0.4, 32.5),
            (-0.7, 40),
            (-8.5, 40),
            (-8.7, 31),
        ],
        PANTS_RGB,
    )
    c.figure_poly(
        [
            (-2.2, -1.2),
            (2.2, -1.2),
            (6.6, 1.0),
            (8.4, 2.4),
            (9.0, 10),
            (8.4, 27.5),
            (-8.4, 27.5),
            (-9.0, 10),
            (-8.4, 2.4),
            (-6.6, 1.0),
        ]
    )
    for s in (1, -1):
        for a, b, w in spec.arms:
            c.limb((a[0] * s, a[1]), (b[0] * s, b[1]), w)
        hx, hy = spec.hand
        c.figure_ellipse(hx * s, hy, 1.15, 1.6)
    c.figure_poly([(-2.0, -4.0), (2.0, -4.0), (2.2, 3.0), (-2.2, 3.0)])
    c.figure_ellipse(0, -6.6, 3.2, 4.1)


def _draw(c: Canvas, t: Template) -> None:
    on_model = t.on_model
    if on_model:
        _draw_figure(c, t)
    if t.view == "folded":
        _draw_folded(c, t)
        return
    spec = t.spec
    if t.garment == "hoodie":
        if t.view == "back":
            c.garment("front", _hood_outer(), _hood_curve())
            c.seam(line((0, -10.3), (0, 1.0)), 200, 0.09)
            _draw_body(c, t, "front")
        else:
            _draw_body(c, t, "front")
            _draw_hoodie_front(c, "back" if on_model else "front", on_model)
        return
    _draw_body(c, t, "front")
    if on_model:
        c.poly("front", _opening(spec), 0)  # the figure's neck shows in the opening


# --- frame and maps -------------------------------------------------------------------------


@lru_cache(maxsize=32)
def product_bbox_in(garment: str, view: str) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) in garment inches, from a coarse draw."""
    t = get_template(garment, view)
    ppi, x0, y0, w, h = 8.0, -30.0, -15.0, 480, 480
    c = Canvas(w, h, ppi, x0, y0, figure=t.on_model)
    _draw(c, t)
    mask = ImageChops.lighter(c.back, c.front) if c.back is not None else c.front
    if c.figure is not None:
        mask = ImageChops.lighter(mask, c.figure)
    left, top, right, bottom = mask.getbbox()
    k = ppi * SS
    box = (left / k + x0, top / k + y0, right / k + x0, bottom / k + y0)
    if t.crop_bottom is not None:
        box = (box[0], box[1], box[2], min(box[3], t.crop_bottom))
    return box


def frame_for(t: Template, w: int, h: int) -> tuple[float, float, float]:
    """(ppi, x0, y0): the product fills FILL of the frame on its limiting side, centered
    (on-model: the bottom edge cuts the figure at `crop_bottom`)."""
    left, top, right, bottom = product_bbox_in(t.garment, t.view)
    bw, bh = right - left, bottom - top
    fill = t.fill
    if t.crop_bottom is not None:
        ppi = min(fill * w / bw, (1 - (1 - fill) / 2) * h / bh)
        x0 = (left + right) / 2 - w / (2 * ppi)
        y0 = bottom - h / ppi
        return ppi, x0, y0
    ppi = min(fill * w / bw, fill * h / bh)
    x0 = (left + right) / 2 - w / (2 * ppi)
    y0 = (top + bottom) / 2 - h / (2 * ppi)
    return ppi, x0, y0


@dataclass
class TemplateMaps:
    template: Template
    width: int
    height: int
    ppi: float
    origin_in: tuple[float, float]  # garment inches at pixel (0, 0)
    back: pyvips.Image | None  # None when the template has nothing behind the figure
    front: pyvips.Image
    shadow: pyvips.Image
    highlight: pyvips.Image
    height_map: pyvips.Image
    figure: pyvips.Image | None
    figure_rgb: pyvips.Image | None
    product: pyvips.Image  # union coverage, 0..255
    print_area_px: tuple[int, int, int, int]
    product_bbox_px: tuple[int, int, int, int]
    flat_px: tuple[int, int] | None

    def to_px(self, x_in: float, y_in: float) -> tuple[float, float]:
        return (x_in - self.origin_in[0]) * self.ppi, (y_in - self.origin_in[1]) * self.ppi


def _vips(im: Image.Image) -> pyvips.Image:
    bands = len(im.getbands())
    # The returned wrapper holds the bytes alive. Don't `copy_memory()` it: on an image that is
    # already in memory that returns the same VipsImage under a wrapper that doesn't hold them.
    return pyvips.Image.new_from_memory(im.tobytes(), im.width, im.height, bands, "uchar")


def _bbox(mask: pyvips.Image) -> tuple[int, int, int, int]:
    x, y, w, h = mask.find_trim(threshold=0.5, background=[0])
    return x, y, w, h


@lru_cache(maxsize=settings.imaging_photo_template_cache)
def template_maps(garment: str, view: str, width: int, height: int) -> TemplateMaps:
    """Draw a template at an output size. Cached: a photo set renders each size many times."""
    t = get_template(garment, view)
    ppi, x0, y0 = frame_for(t, width, height)
    c = Canvas(width, height, ppi, x0, y0, figure=t.on_model)
    _draw(c, t)

    k1 = c.k1

    def down(name: str) -> Image.Image | None:
        """2x coverage -> output size by box averaging: the supersampled antialiasing."""
        im = getattr(c, name)
        if im is None:
            return None
        setattr(c, name, None)  # drop the 2x layer as soon as it's reduced
        return im.reduce(SS) if SS > 1 else im

    back, front, figure = down("back"), down("front"), down("figure")
    figure_rgb = c.figure_rgb
    crisp = c.shadow.filter(ImageFilter.GaussianBlur(0.04 * k1))
    shadow_s = ImageChops.multiply(crisp, c.soft_shadow.filter(ImageFilter.GaussianBlur(0.35 * k1)))
    highlight_s = c.highlight.filter(ImageFilter.GaussianBlur(0.9 * k1))
    height_s = c.soft_height.filter(ImageFilter.GaussianBlur(0.6 * k1))

    product = ImageChops.lighter(back, front) if back is not None else front
    if figure is not None:
        product = ImageChops.lighter(product, figure)

    area = t.print_area
    ax, ay = (area.x - x0) * ppi, (area.y - y0) * ppi
    print_area_px = (round(ax), round(ay), round(area.w * ppi), round(area.h * ppi))
    flat = _find_flat(front, shadow_s, highlight_s, print_area_px)

    pv = _vips(product)
    return TemplateMaps(
        template=t,
        width=width,
        height=height,
        ppi=ppi,
        origin_in=(x0, y0),
        back=_vips(back) if back is not None else None,
        front=_vips(front),
        shadow=_vips(shadow_s),
        highlight=_vips(highlight_s),
        height_map=_vips(height_s),
        figure=_vips(figure) if figure is not None else None,
        figure_rgb=_vips(figure_rgb) if figure_rgb is not None else None,
        product=pv,
        print_area_px=print_area_px,
        product_bbox_px=_bbox(pv),
        flat_px=flat,
    )


def _find_flat(front, shadow, highlight, area) -> tuple[int, int] | None:
    """A fully covered garment pixel with no shading, outside the print area (for tests and
    for anyone checking the blank color). Scans a coarse grid, deterministic."""
    w, h = front.size
    ax, ay, aw, ah = area
    fp, sp, hp = front.load(), shadow.load(), highlight.load()
    step = max(2, min(w, h) // 120)
    pad = 2 * step
    for y in range(h - 1, 0, -step):
        for x in range(0, w, step):
            if ax - pad <= x <= ax + aw + pad and ay - pad <= y <= ay + ah + pad:
                continue
            # The pixel and its neighbours, so a sampler a pixel off still reads the blank.
            if (
                fp[x, y] == 255
                and sp[x, y] == 255
                and hp[x, y] == 0
                and all(
                    fp[x + dx, y + dy] == 255
                    and sp[x + dx, y + dy] == 255
                    and hp[x + dx, y + dy] == 0
                    for dx in (-2, 0, 2)
                    for dy in (-2, 0, 2)
                    if 0 <= x + dx < w and 0 <= y + dy < h
                )
            ):
                return x, y
    return None
