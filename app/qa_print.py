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
8-connected components (pixels that touch, corners included; never across background, so dots
of a halftone stay separate however close they are); a component counts only if it is a *run of
real length* (bounding-box diagonal at least `THIN_RUN_MIN_IN`, 0.10 in). That is what keeps the
tapered tip of a thick shape (a star point, a script letter end, a badge ring edge) from reading
as a thin line: such a tip is under the limit only for its last ~0.03-0.05 in. Its width is the
component's median chord, which makes the reading one analysis pixel accurate for a straight
stroke.

Bounded work (security review S-72/S-73). Memory does not depend on the number of shapes: run
lengths are computed in row blocks, per-component boxes and chord histograms come from libvips
(`hist_find_indexed`, one value per component), and no per-pixel int64 index array is built. At
most `MAX_THIN_COMPONENTS` components are measured. `small_text` is skipped (no issue) when the
art has more than `MAX_TEXT_LABELS` regions or more than `MAX_TEXT_SHAPES` small shapes: that is a
texture or a halftone, not text, and its Python row grouping would otherwise grow without bound.
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
MAX_THIN_COMPONENTS = 300  # the longest ones; only the biggest few hundred matter for the report
MAX_TEXT_LABELS = 1_000_000  # regions (art and holes) in the mask; more is a texture
MAX_TEXT_SHAPES = 5_000  # small shapes of glyph size; more is a halftone or a pattern
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
_RUN_CAP = 255  # run lengths are stored as uint8, clipped; the limits are far below this
_CHUNK_PX = 1 << 20  # run lengths are computed this many pixels at a time
# A chord is coded in one byte: 2*n for a horizontal/vertical run of n pixels, 2*n + 1 for a
# diagonal run of n pixels (n*sqrt 2 long); 0 = not a candidate. Runs are clipped to 100 first.
_CODE_CLIP = 100
_CODE_VALUE = np.array([c / 2 if c % 2 == 0 else (c // 2) * _SQRT2 for c in range(256)])

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
        factor = int(1 / (2 * scale))
        if factor >= 2:
            # A large shrink starts with an integer box shrink: libvips' resize alone on an
            # extreme strip (8,000,000 x 2 px) peaks hundreds of MB higher.
            out_w = max(1, round(rgba.width * scale))
            out_h = max(1, round(rgba.height * scale))
            # Across first: on a very wide strip a vertical pass buffers whole 8M-px lines.
            rgba = rgba.shrinkh(min(factor, rgba.width)).shrinkv(min(factor, rgba.height))
            rgba = rgba.resize(out_w / rgba.width, vscale=out_h / rgba.height, kernel="linear")
        else:
            rgba = rgba.resize(scale, kernel="linear")
    return np.frombuffer(rgba.write_to_memory(), dtype=np.uint8).reshape(rgba.height, rgba.width, 4)


# --- run lengths ------------------------------------------------------------------------------


def _row_runs_block(m: np.ndarray) -> np.ndarray:
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


def _row_runs(m: np.ndarray) -> np.ndarray:
    """For each True pixel, the length of its horizontal run (clipped to uint8); 0 elsewhere.
    Rows are independent, so they are done a block at a time to bound the index arrays."""
    h, w = m.shape
    out = np.empty((h, w), dtype=np.uint8)
    step = max(1, _CHUNK_PX // (w + 1))
    for r in range(0, h, step):
        out[r : r + step] = _row_runs_block(m[r : r + step])
    return out


def _diag_runs(m: np.ndarray) -> np.ndarray:
    """Run lengths along the down-right diagonal (shear each row by its index)."""
    h, w = m.shape
    if h > w:
        return np.ascontiguousarray(_diag_runs(np.ascontiguousarray(m.T)).T)
    stride = w + h
    buf = np.zeros(h * stride, dtype=bool)
    as_strided(buf, shape=(h, w), strides=(stride + 1, 1))[:] = m
    runs = _row_runs(buf.reshape(h, stride))
    del buf
    return as_strided(runs, shape=(h, w), strides=(stride + 1, 1)).copy()


def chord_candidates(mask: np.ndarray, limit_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(candidate bool mask, chord code per pixel as uint8, 0 off the candidates).

    A candidate is a mask pixel with a chord under `limit_px` in some direction; `_CODE_VALUE`
    turns a code back into the chord length in pixels."""
    hv = np.minimum(_row_runs(mask), _row_runs(np.ascontiguousarray(mask.T)).T)
    np.minimum(hv, _CODE_CLIP, out=hv)
    dd = np.minimum(_diag_runs(mask), _diag_runs(np.ascontiguousarray(mask[:, ::-1]))[:, ::-1])
    np.minimum(dd, _CODE_CLIP, out=dd)
    cand = mask & ((hv < limit_px) | (dd < limit_px / _SQRT2))
    # dd * sqrt 2 < hv  <=>  2 dd^2 < hv^2 (exact in uint16 for values up to _CODE_CLIP)
    hv16 = hv.astype(np.uint16)
    dd16 = dd.astype(np.uint16)
    use_d = 2 * dd16 * dd16 < hv16 * hv16
    del hv16, dd16
    code = np.where(use_d, 2 * dd + 1, 2 * hv).astype(np.uint8)
    code[~cand] = 0
    return cand, code


# --- components -------------------------------------------------------------------------------


def _label(binary: np.ndarray) -> np.ndarray:
    """4-connected region labels (uint32) of a bool array; art and background both get labels."""
    h, w = binary.shape
    src = np.ascontiguousarray(binary).view(np.uint8)
    img = pyvips.Image.new_from_memory(src, w, h, 1, "uchar")
    out = img.labelregions()
    return np.frombuffer(out.write_to_memory(), dtype=np.uint32).reshape(h, w)


def _bridge_diagonals(cand: np.ndarray) -> np.ndarray:
    """`cand` plus one pixel in each 2x2 block where two candidates touch only at a corner, so
    4-connected labelling of the result groups `cand` 8-connected. A bridge pixel's other
    neighbours already touch one of the two corners, so nothing across background joins."""
    out = cand.copy()
    out[:-1, 1:] |= cand[:-1, :-1] & cand[1:, 1:]
    out[:-1, :-1] |= cand[:-1, 1:] & cand[1:, :-1]
    return out


def _region_bound(m: np.ndarray) -> int:
    """Upper bound on the 4-connected regions (art and background) of `m`, without labelling:
    the row runs with no pixel of the same value right above them (each region's top row has
    one). Done a block of rows at a time."""
    h, w = m.shape
    total = 0
    step = max(1, _CHUNK_PX // w)
    for r in range(0, h, step):
        blk = m[r : r + step]
        same = np.zeros(blk.shape, dtype=bool)
        if r:
            same[:] = blk == m[r - 1 : r - 1 + blk.shape[0]]
        else:
            same[1:] = blk[1:] == blk[:-1]
        start = np.ones(blk.shape, dtype=bool)
        start[:, 1:] = blk[:, 1:] != blk[:, :-1]
        starts = np.flatnonzero(start)
        total += int(starts.size - np.logical_or.reduceat(same.ravel(), starts).sum())
    return total


def _blocks(sel: np.ndarray, labels: np.ndarray):
    """(rows, ys, xs, labels) of the `sel` pixels, a block of rows at a time (ys within the
    block)."""
    h, w = sel.shape
    step = max(1, _CHUNK_PX // w)
    for r in range(0, h, step):
        ys, xs = np.nonzero(sel[r : r + step])
        if ys.size:
            rows = slice(r, r + step)
            yield rows, ys, xs, labels[rows][ys, xs]


def _stats(labels: np.ndarray, sel: np.ndarray) -> dict[str, np.ndarray]:
    """Per label (index = label): count and bounding box of the pixels where `sel` is True.
    A label without such pixels has count 0 and a meaningless box."""
    n = int(labels.max()) + 1 if labels.size else 1
    big = np.iinfo(np.int32).max
    out = {
        "count": np.zeros(n, dtype=np.int64),
        "x0": np.full(n, big, dtype=np.int32),
        "y0": np.full(n, big, dtype=np.int32),
        "x1": np.zeros(n, dtype=np.int32),
        "y1": np.zeros(n, dtype=np.int32),
    }
    for rows, ys, xs, lab in _blocks(sel, labels):
        out["count"] += np.bincount(lab, minlength=n)
        ys = (ys + rows.start).astype(np.int32)
        xs = xs.astype(np.int32)
        np.minimum.at(out["x0"], lab, xs)
        np.maximum.at(out["x1"], lab, xs)
        np.minimum.at(out["y0"], lab, ys)
        np.maximum.at(out["y1"], lab, ys)
    return out


def _median_chords(labels: np.ndarray, code: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Median chord (pixels) of each label in `keep`, exactly as np.median over its candidate
    pixels, from a per-label histogram of chord codes."""
    slot = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    slot[keep] = np.arange(1, keep.size + 1)
    hist = np.zeros((keep.size + 1) * 256, dtype=np.int64)
    for rows, ys, xs, lab in _blocks(code > 0, labels):
        k = slot[lab].astype(np.int64)
        hist += np.bincount(k * 256 + code[rows][ys, xs], minlength=hist.size)
    counts = hist.reshape(keep.size + 1, 256)[1:]
    order = np.argsort(_CODE_VALUE, kind="stable")
    counts, values = counts[:, order], _CODE_VALUE[order]
    n = counts.sum(axis=1)
    cum = counts.cumsum(axis=1)
    lo = (cum <= ((n - 1) // 2)[:, None]).sum(axis=1)
    hi = (cum <= (n // 2)[:, None]).sum(axis=1)
    return (values[lo] + values[hi]) / 2


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
    cand, code = chord_candidates(mask, warn_px)
    if not cand.any():
        return None
    labels = _label(_bridge_diagonals(cand))
    st = _stats(labels, cand)
    diag = np.hypot(st["x1"] - st["x0"] + 1.0, st["y1"] - st["y0"] + 1.0)
    keep = np.flatnonzero((st["count"] > 0) & (diag >= THIN_RUN_MIN_IN * adpi))
    if not keep.size:
        return None
    keep = keep[np.argsort(-diag[keep], kind="stable")][:MAX_THIN_COMPONENTS]
    medians = _median_chords(labels, code, keep)
    widths: list[float] = []
    regions: list[Region] = []
    for g, median in zip(keep, medians, strict=True):
        if median >= warn_px:
            continue
        widths.append(float(median))
        regions.append(_region(st["x0"][g], st["y0"][g], st["x1"][g], st["y1"][g], w, h))
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
    if _region_bound(mask) > MAX_TEXT_LABELS:
        return None
    labels = _label(mask)
    st = _stats(labels, mask)
    del labels
    x0, y0, x1, y1 = st["x0"], st["y0"], st["x1"], st["y1"]
    gw, gh = x1 - x0 + 1, y1 - y0 + 1
    ok = (st["count"] > 0) & (gh >= min_h) & (gh < max_h) & (gw <= max_w)
    shapes = np.flatnonzero(ok)
    if not TEXT_ROW_MIN <= shapes.size <= MAX_TEXT_SHAPES:
        return None
    cy = (y0[shapes] + y1[shapes]) / 2
    by_y = shapes[np.argsort(cy, kind="stable")]
    bands: list[list[int]] = [[int(by_y[0])]]
    for s in by_y[1:]:
        prev = bands[-1][-1]
        ref = max(gh[s], gh[prev])
        if abs(int(y0[s] + y1[s]) - int(y0[prev] + y1[prev])) / 2 <= 0.35 * ref:
            bands[-1].append(int(s))
        else:
            bands.append([int(s)])
    rows: list[list[int]] = []
    for band in bands:
        if len(band) < TEXT_ROW_MIN:
            continue
        run: list[int] = []
        for s in sorted(band, key=lambda i: x0[i]):
            if run:
                gap = x0[s] - x1[run[-1]] - 1
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
    regions = [_region(min(x0[r]), min(y0[r]), max(x1[r]), max(y1[r]), w, h) for r in rows]
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
