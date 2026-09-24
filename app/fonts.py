"""Bundled OFL fonts (app/fonts) and Pillow text helpers.

Text is drawn with Pillow/FreeType rather than `pyvips.Image.text`: the pyvips binary wheel can
render with a fontfile, but it goes through Pango + fontconfig (which warns about a missing config
in slim containers) and has no stroke support. FreeType gives stable metrics everywhere.
"""

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).parent / "fonts"

# family (lowercase, no spaces) -> (file, variable-font instance name or None)
_FAMILIES: dict[str, tuple[str, str | None]] = {
    "inter": ("Inter.ttf", "Regular"),
    "interbold": ("Inter.ttf", "Bold"),
    "interblack": ("Inter.ttf", "Black"),
    "oswald": ("Oswald.ttf", "Bold"),
    "oswaldregular": ("Oswald.ttf", "Regular"),
    "pacifico": ("Pacifico-Regular.ttf", None),
    "bebasneue": ("BebasNeue-Regular.ttf", None),
    "bebas": ("BebasNeue-Regular.ttf", None),
}

DEFAULT_FAMILY = "inter"
FAMILIES = ("Inter", "Inter Bold", "Inter Black", "Oswald", "Pacifico", "Bebas Neue")


def normalize_family(family: str | None) -> str:
    key = (family or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    return key if key in _FAMILIES else DEFAULT_FAMILY


@lru_cache(maxsize=256)
def font(family: str | None, size_px: int) -> ImageFont.FreeTypeFont:
    file, instance = _FAMILIES[normalize_family(family)]
    f = ImageFont.truetype(str(FONT_DIR / file), max(1, int(size_px)))
    if instance:
        f.set_variation_by_name(instance)
    return f


def text_box(f: ImageFont.FreeTypeFont, text: str, stroke: int = 0) -> tuple[int, int, int, int]:
    """Ink bounding box (left, top, right, bottom) relative to the draw origin."""
    return f.getbbox(text, stroke_width=stroke)


def text_size(f: ImageFont.FreeTypeFont, text: str, stroke: int = 0) -> tuple[int, int]:
    left, top, right, bottom = text_box(f, text, stroke)
    return right - left, bottom - top


def parse_hex(color: str, default: tuple[int, int, int, int] = (0, 0, 0, 255)):
    """'#RGB', '#RRGGBB' or '#RRGGBBAA' -> RGBA tuple. Raises ValueError on garbage."""
    if not color:
        return default
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) not in (6, 8):
        raise ValueError(f"invalid color: {color}")
    vals = [int(c[i : i + 2], 16) for i in range(0, len(c), 2)]
    if len(vals) == 3:
        vals.append(255)
    return tuple(vals)
