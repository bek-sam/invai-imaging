"""Gang-sheet nesting: pack design bounding boxes onto a fixed-width roll.

v1 uses rectangle packing (MaxRects via rectpack) with 90-degree rotation.
Shape-aware nesting (jagua-rs / sparrow) can replace it later.
"""

from dataclasses import dataclass

from rectpack import newPacker


@dataclass(frozen=True)
class Item:
    id: str
    width_in: float
    height_in: float


@dataclass(frozen=True)
class Placement:
    id: str
    x_in: float
    y_in: float
    width_in: float
    height_in: float
    rotated: bool


def nest(
    items: list[Item],
    sheet_width_in: float = 22.0,
    spacing_in: float = 0.25,
    max_length_in: float = 240.0,
    scale: int = 100,
) -> tuple[list[Placement], float]:
    """Place items on a strip `sheet_width_in` wide. Returns placements and used length.

    Sizes are converted to integer units (1/scale inch) because rectpack works best with ints.
    Each item is padded by `spacing_in` so transfers can be cut apart.
    """
    pad = round(spacing_in * scale)
    packer = newPacker(rotation=True)
    for item in items:
        packer.add_rect(round(item.width_in * scale) + pad, round(item.height_in * scale) + pad, item.id)
    packer.add_bin(round(sheet_width_in * scale), round(max_length_in * scale))
    packer.pack()

    by_id = {i.id: i for i in items}
    placements: list[Placement] = []
    length = 0.0
    for _bin, x, y, w, h, rid in packer.rect_list():
        src = by_id[rid]
        rotated = abs((w - pad) / scale - src.width_in) > 1 / scale
        placements.append(
            Placement(rid, x / scale, y / scale, (w - pad) / scale, (h - pad) / scale, rotated)
        )
        length = max(length, (y + h) / scale)

    if len(placements) != len(items):
        raise ValueError("items do not fit within max_length_in; split into more sheets")
    return placements, length


def utilization(items: list[Item], length_in: float, sheet_width_in: float = 22.0) -> float:
    used = sum(i.width_in * i.height_in for i in items)
    return used / (sheet_width_in * length_in) if length_in else 0.0
