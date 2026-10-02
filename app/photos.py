"""Listing photos (T-26-2): the shop's real design on a drawn garment template, at its real print
size, on the exact blank color, sized and checked for one marketplace preset.

- Real size: the design covers exactly `design_width_in x design_height_in` at the template's
  scale (`TemplateMaps.ppi`), horizontally centered on the print area and top-aligned to it (the
  way a presser lines a transfer up under the collar). It is never scaled up; a design larger than
  the print area is scaled down to fit and `design_larger_than_print_area` is reported.
- Real color: the garment is the exact `blank_hex` wherever the template has no shading
  (`TemplateMaps.flat_px` is such a pixel); folds and seams only darken (multiply) or lighten
  (screen) from there, and the print follows the same folds (displaced and shaded).
- Underbase preview: on a dark blank (sRGB luma below UNDERBASE_LUMA) DTF prints white ink under
  the color, so semi-transparent design pixels show over white, not over the blank.
- Output: white background (RGB 255 exactly outside the product), JPEG sRGB (or PNG when
  `out_key` ends in `.png`), optional XMP `dc:subject` keywords, and the preset's checks measured
  on the encoded file.
"""

import re
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import pyvips

from app.fonts import parse_hex
from app.garments import TemplateMaps, template_maps
from app.limits import sniff_format
from app.vips import ImageError, load, to_srgba

# A blank darker than this (Rec. 709 luma of the sRGB values, 0..1) gets the underbase preview.
UNDERBASE_LUMA = 0.45
# Alpha at or above 1/UNDERBASE_GAIN gets full white underbase; fainter pixels get some.
UNDERBASE_GAIN = 4.0
# How far (inches) the steepest fold shifts the print.
FOLD_SHIFT_IN = 0.12
JPEG_QUALITY = 92
JPEG_BLOCK = 8  # 4:4:4 JPEG: ringing stays inside the 8x8 blocks that touch the product


@dataclass(frozen=True)
class Preset:
    width: int
    height: int
    min_longest_px: int
    white_required: bool = False
    min_fill: float | None = None
    photo_required: bool = False  # the marketplace wants a photograph (drawn templates fail)


# Numbers owned by imaging, checked by compliance-officer (wave 26 "Presets").
PRESETS: dict[str, Preset] = {
    "amazon_main": Preset(
        2000, 2000, 1600, white_required=True, min_fill=0.85, photo_required=True
    ),
    "amazon_alt": Preset(2000, 2000, 1000),
    "etsy": Preset(2700, 2025, 2000),  # 4:3, shortest side >= 2000
    "shopify": Preset(2048, 2048, 2048),
    "tiktok": Preset(1600, 1600, 600),
    "walmart": Preset(2000, 2000, 1500, white_required=True),
}


@dataclass
class Checks:
    passes: bool
    failures: list[str]
    background_pure_white: bool | None
    fill_ratio: float
    longest_side_px: int


@dataclass
class PhotoResult:
    data: bytes
    format: str
    width_px: int
    height_px: int
    print_box_px: tuple[int, int, int, int]
    print_area_px: tuple[int, int, int, int]
    scale: float
    underbase_applied: bool
    checks: Checks
    extra: dict = field(default_factory=dict)


def luma(rgb: tuple[int, ...]) -> float:
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255


def output_format(out_key: str) -> str:
    ext = out_key.rsplit(".", 1)[-1].lower() if "." in out_key.rsplit("/", 1)[-1] else ""
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext == "png":
        return "png"
    raise ImageError("out_key must end in .jpg, .jpeg or .png")


def _garment_rgb(maps: TemplateMaps, rgb: tuple[int, int, int]) -> pyvips.Image:
    """The blank, shaded: multiply by the shadow map, then screen the highlight map. Where both
    maps are neutral this is exactly `rgb`."""
    strength = 0.3 + 0.3 * (1 - luma(rgb))  # a softer sheen than /mockup on dark blanks
    shade = maps.shadow.cast("float") / 255
    x = shade * list(rgb)
    hl = maps.highlight.cast("float") * (strength / 255)
    return 255 - (255 - x) * (1 - hl)


def _place(maps: TemplateMaps, w_in: float, h_in: float) -> tuple[int, int, int, int, float]:
    """(x, y, w, h) of the design in output px and the scale applied (<= 1)."""
    area = maps.template.print_area
    scale = min(1.0, area.w / w_in, area.h / h_in)
    dw = max(1, round(w_in * scale * maps.ppi))
    dh = max(1, round(h_in * scale * maps.ppi))
    ax, ay, aw, _ = maps.print_area_px
    return round(ax + (aw - dw) / 2), ay, dw, dh, scale


def _design(path: Path, dw: int, dh: int, dpi: float) -> pyvips.Image:
    """The design resized to exactly dw x dh, premultiplied float RGBA, in memory.

    Raster art goes through `thumbnail` (shrink-on-load, streamed): a 3000 x 3600 px design
    otherwise costs ~130 MB of peak RSS for a ~600 px result. `load()` runs first anyway, so
    the format allowlist, declared-type and pixel-cap checks apply. PDF art (no libvips
    loader) is rasterized at the target DPI by `load()` and resized."""
    img = load(path, access="sequential", target_dpi=max(36.0, dpi))
    if sniff_format(path) in ("png", "jpeg", "webp", "tiff", "svg"):
        img = pyvips.Image.thumbnail(str(path), dw, height=dh, size="force")
    img = to_srgba(img)
    if (img.width, img.height) != (dw, dh):
        img = img.resize(dw / img.width, vscale=dh / img.height, kernel="lanczos3")
    art = img.premultiply()
    if (art.width, art.height) != (dw, dh):  # resize can land a pixel off
        art = art.embed(0, 0, dw, dh, extend="copy")
    return art.copy_memory()


def _shift(height: pyvips.Image, ppi: float) -> tuple[pyvips.Image, pyvips.Image]:
    """Per-pixel print displacement (px) from the fold height map's gradient."""
    h = height.cast("float")
    gx = h.conv(pyvips.Image.new_from_array([[-0.5, 0, 0.5]]), precision="float")
    gy = h.conv(pyvips.Image.new_from_array([[-0.5], [0], [0.5]]), precision="float")
    k = FOLD_SHIFT_IN * ppi * ppi / 70.0  # a 70-level fold over ~1 in shifts FOLD_SHIFT_IN
    return gx * k, gy * k


def compose(
    maps: TemplateMaps,
    design: Path,
    w_in: float,
    h_in: float,
    blank_hex: str,
    underbase_preview: bool,
) -> tuple[pyvips.Image, dict]:
    rgb = tuple(parse_hex(blank_hex)[:3])
    W, H = maps.width, maps.height
    garment = _garment_rgb(maps, rgb).rint().cast("uchar")
    x, y, dw, dh, scale = _place(maps, w_in, h_in)
    underbase = underbase_preview and luma(rgb) < UNDERBASE_LUMA

    # The visible part of the design box.
    px0, py0 = max(0, x), max(0, y)
    px1, py1 = min(W, x + dw), min(H, y + dh)
    front_rgb = garment
    if px1 > px0 and py1 > py0:
        pw, ph = px1 - px0, py1 - py0
        art = _design(design, dw, dh, maps.ppi * scale)
        sx, sy = _shift(maps.height_map.crop(px0, py0, pw, ph), maps.ppi)
        index = pyvips.Image.xyz(pw, ph).cast("float") + [px0 - x, py0 - y]
        index = index - sx.bandjoin(sy)
        art = art.mapim(index).unpremultiply()
        art_rgb, art_a = art[0:3], art[3].cast("float") / 255

        front_a = maps.front.crop(px0, py0, pw, ph).cast("float") / 255
        shade = maps.shadow.crop(px0, py0, pw, ph).cast("float") / 255
        base = garment.crop(px0, py0, pw, ph).cast("float")
        if underbase:
            g = art_a * UNDERBASE_GAIN
            u = (g > 1).ifthenelse(1.0, g)
            white = _garment_rgb_patch(maps, px0, py0, pw, ph)
            base = base * (1 - u) + white * u
        a = art_a * front_a
        ink = art_rgb.cast("float") * (1 - (1 - shade) * 0.5)
        patch = (base * (1 - a) + ink * a).rint().cast("uchar").copy_memory()
        front_rgb = garment.insert(patch, px0, py0)

    # Layer by coverage masks (255 = layer, 0 = below), in 8 bits: exact where a mask is 0 or
    # 255, and far lighter on memory than a float pipeline over the whole frame.
    canvas = (pyvips.Image.black(W, H, bands=3) + 255).cast("uchar")
    if maps.back is not None:
        canvas = maps.back.ifthenelse(garment, canvas, blend=True)
    if maps.figure is not None and maps.figure_rgb is not None:
        canvas = maps.figure.ifthenelse(maps.figure_rgb, canvas, blend=True)
    out = maps.front.ifthenelse(front_rgb, canvas, blend=True).copy(interpretation="srgb")
    return out, {"print_box_px": (x, y, dw, dh), "scale": scale, "underbase": underbase}


def _garment_rgb_patch(maps: TemplateMaps, x: int, y: int, w: int, h: int) -> pyvips.Image:
    """White underbase as it sits on the fabric (same folds), for one patch."""
    shade = maps.shadow.crop(x, y, w, h).cast("float") / 255
    hl = maps.highlight.crop(x, y, w, h).cast("float") * (0.3 / 255)
    white = shade * [255.0, 255.0, 255.0]
    return 255 - (255 - white) * (1 - hl)


# --- encoding and XMP -----------------------------------------------------------------------


def xmp_packet(subjects: list[str]) -> bytes:
    items = "".join(f"<rdf:li>{escape(s)}</rdf:li>" for s in subjects)
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:subject><rdf:Bag>{items}</rdf:Bag></dc:subject>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode()


def _png_with_xmp(png: bytes, xmp: bytes) -> bytes:
    """Insert an iTXt `XML:com.adobe.xmp` chunk right after IHDR (the XMP spec's PNG home).
    libvips' pngsave doesn't write XMP, and Pillow would re-encode the image."""
    data = b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp
    chunk = struct.pack(">I", len(data)) + b"iTXt" + data
    chunk += struct.pack(">I", zlib.crc32(b"iTXt" + data) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature + IHDR length, type, data, crc
    return png[:ihdr_end] + chunk + png[ihdr_end:]


# VipsForeignKeep flags: XMP = 2, ICC = 8.
_KEEP_ICC, _KEEP_XMP = 8, 2


def encode(img: pyvips.Image, fmt: str, xmp_subjects: list[str]) -> bytes:
    xmp = xmp_packet(xmp_subjects) if xmp_subjects else None
    if fmt == "jpeg":
        keep = _KEEP_ICC
        if xmp:
            img = img.copy()
            img.set_type(pyvips.GValue.blob_type, "xmp-data", xmp)
            keep |= _KEEP_XMP
        return img.jpegsave_buffer(
            Q=JPEG_QUALITY, profile="srgb", keep=keep, subsample_mode="off", optimize_coding=True
        )
    png = img.pngsave_buffer(compression=6, profile="srgb", keep=_KEEP_ICC)
    return _png_with_xmp(png, xmp) if xmp else png


# --- checks ---------------------------------------------------------------------------------


def background_is_white(decoded: pyvips.Image, maps: TemplateMaps) -> bool:
    """Every pixel outside the product is exactly RGB 255,255,255 in the encoded file. The
    product region is grown to whole 8x8 JPEG blocks, where compression ringing lives."""
    W, H = maps.width, maps.height
    gw, gh = -(-W // JPEG_BLOCK) * JPEG_BLOCK, -(-H // JPEG_BLOCK) * JPEG_BLOCK
    # Float, so a block with a single faint product pixel doesn't average down to 0.
    blocks = maps.product.cast("float").embed(0, 0, gw, gh).shrink(JPEG_BLOCK, JPEG_BLOCK)
    near = (blocks > 0).zoom(JPEG_BLOCK, JPEG_BLOCK).crop(0, 0, W, H)
    rgb = decoded[0:3] if decoded.bands >= 3 else decoded.bandjoin([decoded, decoded])
    white = rgb.bandbool("and") == 255
    return bool(near.ifthenelse(255, white).min() == 255)


def fill_ratio(maps: TemplateMaps) -> float:
    """Share of the frame the product spans on its limiting side (bounding box)."""
    _, _, bw, bh = maps.product_bbox_px
    return round(max(bw / maps.width, bh / maps.height), 4)


def run_checks(data: bytes, maps: TemplateMaps, preset: Preset, too_big: bool) -> Checks:
    decoded = pyvips.Image.new_from_buffer(data, "")
    longest = max(decoded.width, decoded.height)
    white = background_is_white(decoded, maps)
    fill = fill_ratio(maps)
    failures = []
    if too_big:
        failures.append("design_larger_than_print_area")
    if preset.photo_required and maps.template.drawn:
        failures.append("illustration_not_photo")
    if preset.white_required and not white:
        failures.append("background_not_white")
    if preset.min_fill is not None and fill < preset.min_fill:
        failures.append("fill_below_min")
    if longest < preset.min_longest_px:
        failures.append("too_small")
    return Checks(not failures, failures, white, fill, longest)


def render_photo(
    design: Path,
    w_in: float,
    h_in: float,
    garment: str,
    view: str,
    blank_hex: str,
    underbase_preview: bool,
    preset_name: str,
    fmt: str,
    xmp_subjects: list[str],
) -> PhotoResult:
    preset = PRESETS[preset_name]
    maps = template_maps(garment, view, preset.width, preset.height)
    img, info = compose(maps, design, w_in, h_in, blank_hex, underbase_preview)
    data = encode(img, fmt, xmp_subjects)
    checks = run_checks(data, maps, preset, info["scale"] < 1.0)
    return PhotoResult(
        data=data,
        format=fmt,
        width_px=img.width,
        height_px=img.height,
        print_box_px=info["print_box_px"],
        print_area_px=maps.print_area_px,
        scale=info["scale"],
        underbase_applied=info["underbase"],
        checks=checks,
    )


# --- palette --------------------------------------------------------------------------------

PALETTE_MAX_PX = 160
PALETTE_COLORS = 6
PALETTE_MERGE = 48.0  # RGB distance under which two color bins count as one color


def palette(path: Path) -> dict:
    """Up to 6 dominant colors (alpha-weighted, shares sum to 1 over opaque pixels), plus the
    light (luma >= 0.75), dark (luma <= 0.25) and transparent shares. Deterministic."""
    img = to_srgba(load(path, access="sequential", target_dpi=72))
    scale = min(1.0, PALETTE_MAX_PX / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(scale, kernel="nearest")
    raw = bytes(img.write_to_memory())
    n = img.width * img.height
    bins: dict[tuple[int, int, int], list[float]] = {}
    total_a = light = dark = 0.0
    for i in range(0, 4 * n, 4):
        a = raw[i + 3]
        if not a:
            continue
        r, g, b = raw[i], raw[i + 1], raw[i + 2]
        total_a += a
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        if lum >= 0.75:
            light += a
        elif lum <= 0.25:
            dark += a
        acc = bins.setdefault((r >> 5, g >> 5, b >> 5), [0.0, 0.0, 0.0, 0.0])
        acc[0] += a
        acc[1] += r * a
        acc[2] += g * a
        acc[3] += b * a
    transparent = 1 - total_a / (255 * n) if n else 1.0
    if not total_a:
        return {"colors": [], "light_share": 0.0, "dark_share": 0.0, "transparent_share": 1.0}

    clusters: list[list[float]] = []  # [weight, r, g, b] (means)
    for key in sorted(bins, key=lambda k: (-bins[k][0], k)):
        w, sr, sg, sb = bins[key]
        c = (sr / w, sg / w, sb / w)
        for cl in clusters:
            if _dist(cl, c) < PALETTE_MERGE:
                tw = cl[0] + w
                cl[1:] = [(cl[j + 1] * cl[0] + c[j] * w) / tw for j in range(3)]
                cl[0] = tw
                break
        else:
            clusters.append([w, *c])
    clusters.sort(key=lambda cl: (-cl[0], cl[1], cl[2], cl[3]))
    top, rest = clusters[:PALETTE_COLORS], clusters[PALETTE_COLORS:]
    for cl in rest:  # fold the long tail into the nearest kept color
        min(top, key=lambda t, cl=cl: _dist(t, cl[1:]))[0] += cl[0]
    top.sort(key=lambda cl: (-cl[0], cl[1], cl[2], cl[3]))
    return {
        "colors": [
            {
                "hex": "#{:02X}{:02X}{:02X}".format(*(round(v) for v in cl[1:])),
                "share": round(cl[0] / total_a, 4),
            }
            for cl in top
        ],
        "light_share": round(light / total_a, 4),
        "dark_share": round(dark / total_a, 4),
        "transparent_share": round(transparent, 4),
    }


def _dist(cl, c) -> float:
    a = cl[1:] if len(cl) == 4 else cl
    return ((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 + (a[2] - c[2]) ** 2) ** 0.5


# --- zip ------------------------------------------------------------------------------------


def safe_zip_name(name: str) -> str:
    """A relative path inside the zip: no `..`, no absolute paths, no odd characters."""
    parts = []
    for p in name.replace("\\", "/").split("/"):
        p = re.sub(r"[^\w.\- ]", "_", p).strip(" .")
        if p and p not in (".", ".."):
            parts.append(p[:100])
    return "/".join(parts[-4:]) or "file"


def build_zip(files: list[tuple[Path, str]], dest: Path) -> int:
    """Zip (name, file) pairs; duplicate names get `-2`, `-3`. Already-compressed images are
    stored, everything else deflated. Returns the zip's size in bytes."""
    seen: set[str] = set()
    with zipfile.ZipFile(dest, "w") as zf:
        for path, raw_name in files:
            name = safe_zip_name(raw_name)
            stem, dot, ext = name.rpartition(".")
            if not dot:
                stem, ext = name, ""
            candidate, i = name, 2
            while candidate.lower() in seen:
                candidate = f"{stem}-{i}{'.' + ext if ext else ''}"
                i += 1
            seen.add(candidate.lower())
            stored = candidate.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".zip"))
            zf.write(path, candidate, zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    return dest.stat().st_size
