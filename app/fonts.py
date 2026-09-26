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

FAMILIES = ("Inter", "Inter Bold", "Inter Black", "Oswald", "Pacifico", "Bebas Neue")

# Codepoint ranges (inclusive), within the alphabet `render._ALLOWED` accepts, that each bundled
# font's cmap has no glyph for (Latin Extended-B letters used by a few African/Vietnamese
# orthographies; ordinary Western names are unaffected). Generated once offline from each .ttf
# with fontTools and hardcoded here so imaging carries no fontTools runtime dependency.
_MISSING_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Inter.ttf": ((329, 329), (452, 452)),
    "Oswald.ttf": (
        (384, 398),
        (400, 401),
        (403, 415),
        (418, 430),
        (433, 438),
        (440, 451),
        (463, 466),
        (469, 483),
        (492, 493),
        (496, 496),
        (502, 505),
        (540, 541),
        (544, 553),
        (558, 559),
        (564, 566),
        (568, 591),
    ),
    "Pacifico-Regular.ttf": (
        (306, 307),
        (383, 398),
        (400, 401),
        (403, 415),
        (418, 430),
        (433, 451),
        (461, 485),
        (488, 489),
        (492, 505),
        (540, 553),
        (558, 559),
        (564, 566),
        (568, 591),
    ),
    "BebasNeue-Regular.ttf": (
        (383, 401),
        (403, 507),
        (512, 535),
        (540, 566),
        (568, 591),
    ),
}


def _key(family: str | None) -> str:
    return (family or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def known_family(family: str | None) -> bool:
    """True if `family` (a display name like "Inter Bold") is one of the bundled fonts."""
    return _key(family) in _FAMILIES


def normalize_family(family: str | None) -> str:
    """`family` -> its internal key. Raises ValueError on an unknown name: callers that take a
    family from an API request must validate with `known_family` first (and 422) rather than
    relying on this to silently fall back to Inter, which was the actual bug in B-81."""
    key = _key(family)
    if key not in _FAMILIES:
        raise ValueError(f"unknown font family: {family!r}")
    return key


@lru_cache(maxsize=256)
def font(family: str | None, size_px: int) -> ImageFont.FreeTypeFont:
    file, instance = _FAMILIES[normalize_family(family)]
    f = ImageFont.truetype(str(FONT_DIR / file), max(1, int(size_px)))
    if instance:
        f.set_variation_by_name(instance)
    return f


def missing_chars(family: str | None, text: str) -> list[str]:
    """Characters in `text` (ignoring whitespace) with no glyph in `family`'s bundled font."""
    file, _ = _FAMILIES[normalize_family(family)]
    ranges = _MISSING_RANGES.get(file, ())
    if not ranges:
        return []
    found = {
        ch for ch in text if not ch.isspace() and any(lo <= ord(ch) <= hi for lo, hi in ranges)
    }
    return sorted(found)


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
