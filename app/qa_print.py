"""Print-file checks beyond DPI and alpha (T-42-4, spec print-file-checks): thin lines, small
text, a white box around the art, and a non-sRGB color profile.

All deterministic, no model and no new dependency (numpy and pyvips are already used). Limits are
built in (OI-45 A); there are no request fields for them.

Thin lines. The mask is the visible art (alpha >= 128) at an *analysis resolution*: the file
downscaled so that a pixel is at most ~0.01 in on any sheet-size design (a pixel budget of
`MAX_ANALYSIS_PX` and a cap of `MAX_ANALYSIS_DPI`; a 22 x 60 in design is analysed at ~110 DPI,
0.009 in per pixel). For every mask pixel the *chord* is the shortest run of mask pixels through it
in four directions (horizontal, vertical, both diagonals; diagonal runs are scaled by sqrt 2). A
stroke of width w has chord ~w (within 8% for any angle), and a thick shape has long chords
everywhere except at its corners and tips. Pixels whose chord is under the warn limit form
components; a component counts only if it is a *run of real length* (bounding-box diagonal at least
`THIN_RUN_MIN_IN`, 0.10 in). That is what keeps the tapered tip of a thick shape (a star point,
a script letter end, a badge ring edge) from reading as a thin line: such a tip is under the limit
only for its last ~0.03-0.05 in. Its width is the component's median chord, which makes the
reading one analysis pixel accurate for a straight stroke.
"""

import math
import struct

import numpy as np
import pyvips
from numpy.lib.stride_tricks import as_strided

THIN_LINE_ERROR_IN = 0.02
THIN_LINE_WARN_IN = 0.03
SMALL_TEXT_IN = 0.10
NEAR_WHITE = 245
WHITE_RING_SHARE = 0.95

THIN_RUN_MIN_IN = 0.10  # a thin component shorter than this is a tip or a speck, not a line
MAX_ANALYSIS_PX = 16_000_000
MAX_ANALYSIS_DPI = 300
MAX_REGIONS = 20
ALPHA_VISIBLE = 128  # alpha >= this is "ink" at native resolution
# After a shrink a stroke thinner than a pixel is spread over two pixels at about half coverage
# each, so a threshold of 50% would lose it. 30% keeps it and overstates a stroke by under half
# a pixel.
ALPHA_VISIBLE_SHRUNK = 77
# small_text: a shape counts if it is between these heights, no wider than 3x SMALL_TEXT_IN, and
# four or more of them sit on one line with gaps of at most 2.5x their height.
TEXT_MIN_IN = 0.02
TEXT_MAX_WIDTH_IN = 0.30
TEXT_ROW_MIN = 4
_SQRT2 = math.sqrt(2.0)
_RUN_CAP = 255  # chords are stored as uint8, clipped; the limits are far below this

Region = dict[str, float]


# --- analysis image ---------------------------------------------------------------------------


def analysis_scale(width_px: int, height_px: int, eff_dpi: float | None) -> float:
    """Shrink factor (<= 1) for the analysis image."""
    s = math.sqrt(MAX_ANALYSIS_PX / (width_px * height_px))
    if eff_dpi:
        s = min(s, MAX_ANALYSIS_DPI / eff_dpi)
    return min(1.0, s)


def analysis_array(rgba: pyvips.Image, scale: float) -> np.ndarray:
    """The sRGBA image shrunk by `scale` (alpha-aware) as an HxWx4 uint8 array."""
    if scale < 1.0:
        rgba = rgba.resize(scale, kernel="linear")
    return np.frombuffer(rgba.write_to_memory(), dtype=np.uint8).reshape(rgba.height, rgba.width, 4)


# --- run lengths ------------------------------------------------------------------------------


def _row_runs(m: np.ndarray) -> np.ndarray:
    """For each True pixel, the length of its horizontal run (clipped to uint8); 0 elsewhere."""
    h, w = m.shape
    padded = np.zeros((h, w + 1), dtype=bool)
    padded[:, :w] = m
    flat = padded.ravel()
    d = np.diff(flat.view(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(d == 1)
    lengths = np.flatnonzero(d == -1) - starts
    out = np.zeros(flat.size, dtype=np.uint8)
    out[flat] = np.repeat(np.minimum(lengths, _RUN_CAP).astype(np.uint8), lengths)
    return out.reshape(h, w + 1)[:, :w]


def _diag_runs(m: np.ndarray) -> np.ndarray:
    """Run lengths along the down-right diagonal (shear each row by its index)."""
    h, w = m.shape
    if h > w:
        return np.ascontiguousarray(_diag_runs(np.ascontiguousarray(m.T)).T)
    stride = w + h
    buf = np.zeros(h * stride, dtype=bool)
    as_strided(buf, shape=(h, w), strides=(stride + 1, 1))[:] = m
    runs = _row_runs(buf.reshape(h, stride))
    ys = np.arange(h)[:, None]
    return runs[ys, np.arange(w)[None, :] + ys]


def chord_candidates(mask: np.ndarray, limit_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(candidate bool mask, chord length in pixels for the candidate pixels in nonzero order).

    A candidate is a mask pixel with a chord under `limit_px` in some direction."""
    hv = np.minimum(_row_runs(mask), _row_runs(np.ascontiguousarray(mask.T)).T)
    dd = np.minimum(_diag_runs(mask), _diag_runs(np.ascontiguousarray(mask[:, ::-1]))[:, ::-1])
    cand = mask & ((hv < limit_px) | (dd * _SQRT2 < limit_px))
    ys, xs = np.nonzero(cand)
    chord = np.minimum(hv[ys, xs].astype(np.float32), dd[ys, xs].astype(np.float32) * _SQRT2)
    return cand, chord


# --- components -------------------------------------------------------------------------------


def _label(binary: np.ndarray) -> np.ndarray:
    """Connected-component labels (uint32) of a bool array; label values of False pixels are
    meaningless and never read."""
    h, w = binary.shape
    img = pyvips.Image.new_from_memory((binary.view(np.uint8) * 255).tobytes(), w, h, 1, "uchar")
    out = img.labelregions()
    return np.frombuffer(out.write_to_memory(), dtype=np.uint32).reshape(h, w)


def _boxes(labels: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-label bounding boxes of the pixels (ys, xs) carrying `labels` (same length)."""
    order = np.argsort(labels, kind="stable")
    lab, y, x = labels[order], ys[order], xs[order]
    first = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    return {
        "order": order,
        "first": first,
        "count": np.diff(np.r_[first, lab.size]),
        "x0": np.minimum.reduceat(x, first),
        "x1": np.maximum.reduceat(x, first),
        "y0": np.minimum.reduceat(y, first),
        "y1": np.maximum.reduceat(y, first),
    }


def _region(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> Region:
    """Pixel box (inclusive x1, y1) as ratios of the image, clamped to 0..1."""

    def c(v: float) -> float:
        return round(float(min(1.0, max(0.0, v))), 5)

    return {
        "left": c(x0 / w),
        "top": c(y0 / h),
        "width": c((x1 + 1 - x0) / w),
        "height": c((y1 + 1 - y0) / h),
    }


def merge_regions(regions: list[Region], limit: int = MAX_REGIONS) -> list[Region]:
    """Union overlapping or touching boxes, then largest first, at most `limit`."""
    boxes = [
        [r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"]]
        for r in sorted(regions, key=lambda r: -(r["width"] * r["height"]))[:300]
    ]
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] and o[0] <= b[2] and b[1] <= o[3] and o[1] <= b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    boxes.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    return [
        {
            "left": round(b[0], 5),
            "top": round(b[1], 5),
            "width": round(b[2] - b[0], 5),
            "height": round(b[3] - b[1], 5),
        }
        for b in boxes[:limit]
    ]


# --- thin lines -------------------------------------------------------------------------------


def thin_lines(mask: np.ndarray, adpi: float) -> tuple[str, float, list[Region]] | None:
    """(severity, thinnest stroke in inches, regions) or None. `adpi` = analysis pixels per inch."""
    h, w = mask.shape
    warn_px = THIN_LINE_WARN_IN * adpi - 1e-6
    error_px = THIN_LINE_ERROR_IN * adpi - 1e-6
    cand, chord = chord_candidates(mask, warn_px)
    if not chord.size:
        return None
    # Dilate by one pixel before labelling so a one-pixel diagonal stays one component.
    dil = cand.copy()
    dil[1:, :] |= cand[:-1, :]
    dil[:-1, :] |= cand[1:, :]
    dil[:, 1:] |= dil[:, :-1].copy()
    dil[:, :-1] |= dil[:, 1:].copy()
    labels = _label(dil)
    ys, xs = np.nonzero(cand)
    lab = labels[ys, xs]
    b = _boxes(lab, ys, xs)
    diag = np.hypot(b["x1"] - b["x0"] + 1, b["y1"] - b["y0"] + 1)
    keep = np.flatnonzero(diag >= THIN_RUN_MIN_IN * adpi)
    if not keep.size:
        return None
    # The widest components first: only the biggest few hundred matter for the report.
    keep = keep[np.argsort(-diag[keep])][:300]
    sorted_chord = chord[b["order"]]
    widths: list[float] = []
    regions: list[Region] = []
    for g in keep:
        start = b["first"][g]
        median = float(np.median(sorted_chord[start : start + b["count"][g]]))
        if median >= warn_px:
            continue
        widths.append(median)
        regions.append(_region(b["x0"][g], b["y0"][g], b["x1"][g], b["y1"][g], w, h))
    if not widths:
        return None
    thinnest = min(widths)
    severity = "error" if thinnest < error_px else "warn"
    return severity, thinnest / adpi, merge_regions(regions)


# --- small text -------------------------------------------------------------------------------


def small_text(mask: np.ndarray, adpi: float) -> tuple[float, list[Region]] | None:
    """(median shape height in inches, regions) of rows of small similar shapes, or None."""
    h, w = mask.shape
    max_h = SMALL_TEXT_IN * adpi
    max_w = TEXT_MAX_WIDTH_IN * adpi
    min_h = max(2.0, TEXT_MIN_IN * adpi)
    labels = _label(mask)
    counts = np.bincount(labels[mask])
    # A glyph fits in max_h x max_w; anything with more pixels is not a small shape.
    small = counts <= max_h * max_w
    ys, xs = np.nonzero(mask & small[np.minimum(labels, counts.size - 1)])
    if not ys.size:
        return None
    lab = labels[ys, xs]
    b = _boxes(lab, ys, xs)
    gw, gh = b["x1"] - b["x0"] + 1, b["y1"] - b["y0"] + 1
    ok = (gh >= min_h) & (gh < max_h) & (gw <= max_w)
    shapes = np.flatnonzero(ok)
    if shapes.size < TEXT_ROW_MIN:
        return None
    cy = (b["y0"][shapes] + b["y1"][shapes]) / 2
    by_y = shapes[np.argsort(cy, kind="stable")]
    bands: list[list[int]] = [[int(by_y[0])]]
    for s in by_y[1:]:
        prev = bands[-1][-1]
        ref = max(gh[s], gh[prev])
        if abs((b["y0"][s] + b["y1"][s]) - (b["y0"][prev] + b["y1"][prev])) / 2 <= 0.35 * ref:
            bands[-1].append(int(s))
        else:
            bands.append([int(s)])
    rows: list[list[int]] = []
    for band in bands:
        if len(band) < TEXT_ROW_MIN:
            continue
        run: list[int] = []
        for s in sorted(band, key=lambda i: b["x0"][i]):
            if run:
                gap = b["x0"][s] - b["x1"][run[-1]] - 1
                ratio = gh[s] / float(np.median(gh[run]))
                if gap > 2.5 * max(gh[s], gh[run[-1]]) or not 0.4 <= ratio <= 2.5:
                    if len(run) >= TEXT_ROW_MIN:
                        rows.append(run)
                    run = []
            run.append(s)
        if len(run) >= TEXT_ROW_MIN:
            rows.append(run)
    if not rows:
        return None
    heights = np.concatenate([gh[r] for r in rows])
    regions = [
        _region(min(b["x0"][r]), min(b["y0"][r]), max(b["x1"][r]), max(b["y1"][r]), w, h)
        for r in rows
    ]
    return float(np.median(heights)) / adpi, merge_regions(regions)


# --- white box --------------------------------------------------------------------------------


def white_box(rgba: np.ndarray) -> list[Region] | None:
    """Regions of the outer ring when >= WHITE_RING_SHARE of it is opaque near-white, else None.

    Only call this for a file that has transparency somewhere."""
    h, w, _ = rgba.shape
    ring = max(2, round(0.01 * min(h, w)))
    if 2 * ring >= min(h, w):
        return None
    strips = {
        "top": rgba[:ring, :, :],
        "bottom": rgba[h - ring :, :, :],
        "left": rgba[ring : h - ring, :ring, :],
        "right": rgba[ring : h - ring, w - ring :, :],
    }
    total = white = 0
    for s in strips.values():
        total += s.shape[0] * s.shape[1]
        is_white = (s[..., 3] >= 254) & (s[..., :3].min(axis=2) >= NEAR_WHITE)
        white += int(is_white.sum())
    if white / total < WHITE_RING_SHARE:
        return None
    return [
        _region(0, 0, w - 1, ring - 1, w, h),
        _region(0, h - ring, w - 1, h - 1, w, h),
        _region(0, ring, ring - 1, h - ring - 1, w, h),
        _region(w - ring, ring, w - 1, h - ring - 1, w, h),
    ]


# --- color profile ----------------------------------------------------------------------------

# sRGB colorants, D50-adapted (ICC profile connection space): the primaries of an sRGB profile.
_SRGB_XYZ = {
    b"rXYZ": (0.4360, 0.2225, 0.0139),
    b"gXYZ": (0.3851, 0.7169, 0.0971),
    b"bXYZ": (0.1431, 0.0606, 0.7141),
}
_XYZ_TOLERANCE = 0.02


def _profile_is_srgb(data: bytes) -> bool | None:
    """True/False for an RGB profile whose colorants can be read; None if unknown."""
    try:
        count = struct.unpack(">I", data[128:132])[0]
        tags = {}
        for i in range(min(count, 200)):
            sig, off, size = struct.unpack(">4sII", data[132 + 12 * i : 144 + 12 * i])
            tags[sig] = (off, size)
        if not all(k in tags for k in _SRGB_XYZ):
            return None
        for sig, ref in _SRGB_XYZ.items():
            off, _ = tags[sig]
            xyz = struct.unpack(">3i", data[off + 8 : off + 20])
            if any(abs(v / 65536 - r) > _XYZ_TOLERANCE for v, r in zip(xyz, ref, strict=True)):
                return False
        return True
    except (struct.error, IndexError):
        return None


def is_non_srgb(img: pyvips.Image) -> bool:
    """True for CMYK mode, a CMYK profile, or an RGB profile that is not sRGB. Call on the file as
    loaded, before any conversion. No profile counts as sRGB; a gray profile is ignored."""
    if img.interpretation == "cmyk":
        return True
    if img.get_typeof("icc-profile-data") == 0:
        return False
    data = bytes(img.get("icc-profile-data"))
    space = data[16:20]
    if space == b"CMYK":
        return True
    if space != b"RGB ":
        return False
    return _profile_is_srgb(data) is False
