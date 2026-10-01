"""Gang-sheet nesting: pack design boxes onto a fixed-width roll.

MaxRects (rectpack's free-rectangle bookkeeping) with our own placement choice, because a design's
footprint is not a plain rectangle to rotate: the order label strip always sits *under* the placed
design, so a rotated design of w x h occupies (h, w + label), not (h + label, w).

Footprint of a design placed as pw x ph:  (pw + spacing) x (ph + label + spacing).
The bin is (usable width + spacing) wide so the last column needs no trailing gap, and margins are
kept clear on all four sides. Several heuristics and orderings are tried per sheet and the one that
places the most area in the shortest length wins. Items that don't fit go to the next sheet.
"""

from dataclasses import dataclass, field

from rectpack.geometry import Rectangle
from rectpack.maxrects import MaxRects

SCALE = 1000  # work in mils so rectpack deals with integers


@dataclass(frozen=True)
class Item:
    id: str
    width_in: float
    height_in: float
    quantity: int = 1


@dataclass(frozen=True)
class Placement:
    id: str
    copy: int
    x_in: float
    y_in: float
    width_in: float
    height_in: float
    rotated: bool


@dataclass
class Sheet:
    index: int
    length_in: float
    utilization: float
    placements: list[Placement] = field(default_factory=list)


class NestError(ValueError):
    pass


@dataclass(frozen=True)
class _Piece:
    id: str
    copy: int
    w: int  # design size in mils, unrotated
    h: int


class _Bin(MaxRects):
    def __init__(self, width: int, height: int):
        super().__init__(width, height, rot=False)

    def place(self, options, heuristic: str):
        """options: [(fw, fh, rotated)]. Returns (Rectangle, rotated) or None."""
        best = None
        for m in self._max_rects:
            for fw, fh, rotated in options:
                if fw > m.width or fh > m.height:
                    continue
                if heuristic == "bl":
                    score = (m.y + fh, m.x)
                elif heuristic == "bssf":
                    score = (min(m.width - fw, m.height - fh), m.y + fh)
                else:  # "baf"
                    score = (m.width * m.height - fw * fh, m.y + fh)
                if best is None or score < best[0]:
                    best = (score, Rectangle(m.x, m.y, fw, fh), rotated)
        if best is None:
            return None
        rect = best[1]
        self._split(rect)
        self._remove_duplicates()
        self.rectangles.append(rect)
        return rect, best[2]


_ORDERS = {
    "height": lambda p: (-max(p.w, p.h), -p.w * p.h),
    "area": lambda p: (-p.w * p.h, -max(p.w, p.h)),
    "width": lambda p: (-p.w, -p.h),
}


def _mils(v: float) -> int:
    return round(v * SCALE)


# How a piece may be oriented: both ways, never rotated, or pre-rotated so that the most copies
# fit across the width (a 10.5 x 12 in front goes two-up on 22 in film; rotated it goes one-up).
_ROTATIONS = ("free", "none", "across")


def _options(p, bin_w, spacing, label, rotation):
    upright = (p.w + spacing, p.h + label + spacing, False)
    if rotation == "none" or p.w == p.h:
        return [upright]
    turned = (p.h + spacing, p.w + label + spacing, True)
    if rotation == "free":
        return [upright, turned]
    across_upright, across_turned = bin_w // upright[0], bin_w // turned[0]
    if across_turned > across_upright:
        return [turned]
    if across_upright > across_turned:
        return [upright]
    return [upright, turned]


def _pack_one(pieces, bin_w, bin_h, spacing, label, rotation, heuristic):
    b = _Bin(bin_w, bin_h)
    placed, rest = [], []
    for p in pieces:
        options = _options(p, bin_w, spacing, label, rotation)
        res = b.place(options, heuristic)
        if res is None:
            rest.append(p)
        else:
            rect, rotated = res
            placed.append((p, rect, rotated))
    return placed, rest


def nest(
    items: list[Item],
    sheet_width_in: float = 22.0,
    spacing_in: float = 0.25,
    margin_in: float = 0.25,
    max_length_in: float = 240.0,
    allow_rotation: bool = True,
    label_height_in: float = 0.42,
    header_height_in: float = 0.45,
) -> list[Sheet]:
    if sheet_width_in <= 0 or max_length_in <= 0:
        raise NestError("sheet_width_in and max_length_in must be positive")
    if spacing_in < 0 or margin_in < 0 or label_height_in < 0 or header_height_in < 0:
        raise NestError(
            "spacing_in, margin_in, label_height_in and header_height_in must not be negative"
        )

    spacing, label, margin = _mils(spacing_in), _mils(label_height_in), _mils(margin_in)
    header = _mils(header_height_in)
    # The header strip is a fixed top-edge band, above the margin: nesting starts below both.
    usable_w = _mils(sheet_width_in) - 2 * margin
    usable_h = _mils(max_length_in) - 2 * margin - header
    if usable_w <= 0 or usable_h <= 0:
        raise NestError("margins and header leave no usable area")
    bin_w, bin_h = usable_w + spacing, usable_h + spacing

    pieces: list[_Piece] = []
    for it in items:
        if it.width_in <= 0 or it.height_in <= 0:
            raise NestError(f"item {it.id}: width_in and height_in must be positive")
        if it.quantity < 1:
            continue
        w, h = _mils(it.width_in), _mils(it.height_in)
        fits = w <= usable_w and h + label <= usable_h
        if allow_rotation:
            fits = fits or (h <= usable_w and w + label <= usable_h)
        if not fits:
            raise NestError(
                f"item {it.id} ({it.width_in}x{it.height_in}in) does not fit a "
                f"{sheet_width_in}x{max_length_in}in sheet"
            )
        pieces.extend(_Piece(it.id, c, w, h) for c in range(it.quantity))

    sheets: list[Sheet] = []
    remaining = pieces
    while remaining:
        best = None
        rotations = _ROTATIONS if allow_rotation else ("none",)
        for order in _ORDERS.values():
            ordered = sorted(remaining, key=order)
            for rotation in rotations:
                for heuristic in ("bl", "bssf", "baf"):
                    placed, rest = _pack_one(
                        ordered, bin_w, bin_h, spacing, label, rotation, heuristic
                    )
                    area = sum(p.w * p.h for p, _, _ in placed)
                    used = max((r.y + r.height for _, r, _ in placed), default=0)
                    score = (-area, used)
                    if best is None or score < best[0]:
                        best = (score, placed, rest)
        _, placed, remaining = best
        sheets.append(_to_sheet(len(sheets), placed, sheet_width_in, spacing, margin, header))
    return sheets


def overall_utilization(sheets: list[Sheet]) -> float:
    """Length-weighted film use across the whole sheet set (B-41): total design area over total
    film used, not a plain average of each sheet's own utilization -- a lightly filled last sheet
    pulls this down by how much film it actually used, rather than counting as much as a full
    one. Each sheet's `utilization` is area_i / (width * length_i), so weighting by length_in and
    dividing by the total length recovers (sum of area_i) / (width * total length) without
    needing the width again."""
    total_length = sum(s.length_in for s in sheets)
    if total_length <= 0:
        return 0.0
    return round(sum(s.utilization * s.length_in for s in sheets) / total_length, 4)


def _to_sheet(index, placed, sheet_width_in, spacing, margin, header) -> Sheet:
    placements = []
    used = 0
    area = 0
    for p, r, rotated in sorted(placed, key=lambda t: (t[1].y, t[1].x)):
        pw, ph = (p.h, p.w) if rotated else (p.w, p.h)
        placements.append(
            Placement(
                id=p.id,
                copy=p.copy,
                x_in=(margin + r.x) / SCALE,
                y_in=(margin + header + r.y) / SCALE,
                width_in=pw / SCALE,
                height_in=ph / SCALE,
                rotated=rotated,
            )
        )
        used = max(used, r.y + r.height - spacing)
        area += p.w * p.h
    length = (used + 2 * margin + header) / SCALE
    utilization = area / (SCALE * SCALE) / (sheet_width_in * length) if length else 0.0
    return Sheet(index, round(length, 3), round(utilization, 4), placements)
