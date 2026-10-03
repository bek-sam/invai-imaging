"""AI lifestyle scenes with the design locked (T-27-2, ADR 0023 phase B).

`/photo/scene-base` draws a blank garment (no design) for the image provider, with an edit mask
in OpenAI's convention: a PNG with alpha, transparent = the provider may draw, opaque = keep.
Only the print area plus MASK_MARGIN_IN is protected.

`/photo/scene-composite` takes the provider's scene back. GPT Image treats the mask as guidance
only and answers 1024x1024, 1536x1024 or 1024x1536, so nothing about the scene is trusted:
1. Registration: the scene is cover-fitted to the base frame, then a scale + translation is
   searched (coarse to fine, normalized cross-correlation of edge maps over the garment and
   figure) that maps base pixels onto it.
2. Check 1, `region_changed` ("garment outline aligned"): after registration, the edge map of the
   scene is compared with the base's in a band along the garment outline, inside a ring around
   the print box. A moved or reshaped garment scores low; no composite is saved.
3. The provider frame is warped (registration and upscale in one resample, `lbb` interpolation:
   locally bounded bicubic, no halos) straight to the preset size and cropped to its aspect.
4. The base's blank print-box pixels are restored (the template redrawn at output size, so the
   blank is exactly `blank_hex` wherever it isn't shaded), lit by the scene: a global gain and a
   local shade, both from the scene's luminance relative to the blank. The design is composited
   at real size, displaced and shaded by the same light, with the wave 26 underbase rule.
5. Check 2, `design_drift`: the printed region is de-shaded (smooth achromatic luminance
   normalization) and compared with the source design flat on the blank, by SSIM.
"""

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvips

from app import photos
from app.fonts import parse_hex
from app.garments import SCENE_VIEWS, TemplateMaps, template_maps
from app.limits import pixel_budget_error, sniff_format
from app.vips import ImageError, load

# --- base -----------------------------------------------------------------------------------

# Protected margin around the print area in the edit mask, and the width (inches) of the band
# around the print area that is restored from the base before compositing (same value: the
# restored band is exactly the region the provider was asked to keep).
MASK_MARGIN_IN = 0.5
# Neutral set behind the figure on `lifestyle_base`: the provider replaces it with a scene.
LIFESTYLE_BG = (228, 226, 222)
BASE_META_KEY = b"invai:scene-base"

# --- scene input (untrusted model output) ---------------------------------------------------

SCENE_FORMATS = ("png", "jpeg", "webp")
SCENE_MIN_SIDE = 256
SCENE_MAX_SIDE = 4096  # GPT Image answers at most 1536; 4096 leaves room for other providers
SCENE_MAX_BYTES = 25 * 1024 * 1024

# --- registration ---------------------------------------------------------------------------

REG_COARSE_PX = 256  # longest side of the coarse search
REG_FINE_PX = 768  # longest side of the fine search
REG_MAX_SHIFT = 0.08  # translation searched, share of the longest side
REG_SCALES = tuple(round(0.90 + 0.02 * i, 2) for i in range(11))  # 0.90 .. 1.10
REG_FINE_STEP = 0.005
EDGE_KNEE = 6.0  # gradient (levels/px) at which an edge counts half

# --- check 1: garment outline aligned -------------------------------------------------------

RING_IN = 6.5  # ring around the print box: reaches the collar and side seams on every garment
CHECK1_PX = 1536  # longest side check 1 runs at
BAND_IN = 0.35  # tolerance band along the garment outline where edges are compared
# Measured (README "Scene lock", 32 scenes: tee and hoodie, on_model_white and lifestyle_base,
# navy and white): unchanged and mock-like perturbed scenes (new background, relit and tinted,
# noise, 2% scale, shift, provider size, JPEG) and scenes with strong folds score 0.92-1.00; a
# garment widened 10% scores 0.01-0.15; a frame shifted 15% scores <= 0.33 and hits the search
# limit. 0.75 leaves 0.17 below the lowest pass and 0.42 above the highest fail.
REGION_UNCHANGED_MIN = 0.75

# --- check 1b: nothing drawn into the print area ------------------------------------------------

BOX_PX = 256  # longest side of the print area this check runs at (about 0.03-0.05 in per px)
BOX_BLUR_PX = 1.0  # sensor noise and JPEG blocks blurred away before the gradient
BOX_LUMA_FLOOR = 8.0  # keeps log contrast finite on near-black fabric
BOX_EDGE_MIN = 0.10  # log-luminance step per px that counts as a drawn edge
# Measured (7 garment/view/blank cases incl. black and red, JPEG q85): unchanged, mock-like
# perturbed, provider-size and +-15%/+-30% fold scenes score 0.000 at 0.10; provider art drawn
# into the box with the outline intact (the reviewer's disc + bars) 0.068-0.092, a small
# three-line wordmark in one corner 0.012-0.027. 0.004 (about 260 px of 65k) sits well below
# the smallest mark and above every clean scene.
BOX_FOREIGN_MAX = 0.004

# --- composite ------------------------------------------------------------------------------

# Scene light carried onto the restored blank and the print: low frequency only (broad folds
# and light), never detail, so nothing the provider drew there can reach the output even below
# check 1b's threshold (a 0.33 in bar keeps < 10% of its contrast at this blur), and bounded.
SHADE_SIGMA_IN = 0.4
SHADE_RANGE = (0.7, 1.2)  # local light relative to the blank
GAIN_RANGE = (0.5, 1.6)  # global light of the scene's garment relative to the base
GAIN_TINT = 0.15  # most a channel's gain may differ from the luminance gain (white balance)
HEIGHT_SIGMA_IN = 0.3  # the scene's light becomes a height map for displacement at this scale
# Scene light read as relief: half of a template fold's height per unit of shadow (a template
# fold is 70 height levels for a 19% shadow), so the print follows the scene's folds gently.
FOLD_HEIGHT_PER_SHADE = 0.5 * 70 / 0.19

# --- check 2: design lock -------------------------------------------------------------------

SSIM_PX = 160  # longest side the printed region is compared at (a few px of fold shift vanish)
SSIM_SIGMA = 1.5
DESHADE_SIGMA = 0.06  # share of the region's longest side: smooth light and broad folds
DESHADE_RANGE = (0.6, 1.6)
# Measured on the same 32 scenes: the design printed into unchanged or perturbed scenes scores
# 0.94-1.00, into scenes with +-30% folds over the print 0.88-0.90; a swapped design 0.36-0.42,
# the same design with its main color changed 0.69-0.71. 0.80 leaves 0.08 below the lowest pass
# and 0.09 above the highest fail.
DESIGN_LOCK_MIN = 0.80


class SceneInputError(Exception):
    """The provider's scene is not an image we accept (format, size, bytes). HTTP 400."""


# --- helpers --------------------------------------------------------------------------------


def _luma(img: pyvips.Image) -> pyvips.Image:
    return img[0:3].cast("float").recomb([[0.2126, 0.7152, 0.0722]])


def _warp(img: pyvips.Image, m: float, off: tuple[float, float], w: int, h: int) -> pyvips.Image:
    """out = m * in + off in corner coordinates (pixel (0,0) spans 0..1), output w x h.
    libvips' affine maps pixel indices, hence the half-pixel terms. Output pixels that fall
    outside the input repeat its edge (affine alone fills them black, which reads as an edge)."""
    pad = max(8, math.ceil(max(w, h) / max(m, 1e-3) * 0.25))
    src = img.embed(pad, pad, img.width + 2 * pad, img.height + 2 * pad, extend="copy")
    return src.affine(
        [m, 0, 0, m],
        oarea=[0, 0, w, h],
        odx=off[0] - m * pad + 0.5 * m - 0.5,
        ody=off[1] - m * pad + 0.5 * m - 0.5,
        interpolate=pyvips.Interpolate.new("lbb" if m > 1 else "bicubic"),
        extend="copy",
    )


def _edges(
    img: pyvips.Image, sigma: float = 1.0, post: float = 1.5, compress: bool = True
) -> np.ndarray:
    """Gradient magnitude of luminance, softly saturated (a relit or recolored scene changes edge
    contrast: m / (m + EDGE_KNEE) counts an edge as an edge whatever its contrast, while sensor
    noise stays near 0) and blurred by `post` so the correlation surface is smooth enough to
    search."""
    lum = _luma(img).gaussblur(sigma, precision="float").numpy().astype(np.float32)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    mag = mag / (mag + EDGE_KNEE) if compress else mag
    return _gauss(mag, post) if post > 0 else mag


def _fast_len(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n: FFT sizes pocketfft handles fast."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


class _Template:
    """A registration template (edge map `t`, weights `w`) with its spectra cached per FFT size,
    so each scale tried costs two forward FFTs of the scene window and three inverse ones."""

    def __init__(self, t: np.ndarray, w: np.ndarray) -> None:
        self.t, self.w = t.astype(np.float32), w.astype(np.float32)
        self.sw = float(self.w.sum()) or 1.0
        tz = self.t - (self.t * self.w).sum() / self.sw
        self.tzw = tz * self.w
        self.tn = math.sqrt(float((self.tzw * tz).sum())) or 1.0
        self._spec: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def spectra(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if shape not in self._spec:
            self._spec[shape] = (
                np.fft.rfft2(self.tzw[::-1, ::-1], shape),
                np.fft.rfft2(self.w[::-1, ::-1], shape),
            )
        return self._spec[shape]


def _ncc_search(e: np.ndarray, tpl: _Template, x0: int, y0: int, r: int) -> np.ndarray:
    """Masked NCC (weights `tpl.w`) of template `tpl.t` against `e` with its top-left at
    (x0 + dx, y0 + dy), dx, dy in [-r, r]. Returns a (2r+1, 2r+1) surface indexed
    [dy + r, dx + r]. Weighting leaves out the background inside the product's bounding box,
    which the provider is free to redraw. Correlations by FFT."""
    th, tw = tpl.t.shape
    pad = np.pad(e, r + 1, mode="edge")
    sub = pad[y0 + 1 : y0 + 1 + th + 2 * r, x0 + 1 : x0 + 1 + tw + 2 * r]
    shape = (_fast_len(sub.shape[0]), _fast_len(sub.shape[1]))
    ft, fw = tpl.spectra(shape)
    fa, fa2 = np.fft.rfft2(sub, shape), np.fft.rfft2(sub * sub, shape)
    n = 2 * r + 1

    def valid(spec):
        return np.fft.irfft2(spec, shape)[th - 1 : th - 1 + n, tw - 1 : tw - 1 + n]

    num, s1, s2 = valid(fa * ft), valid(fa * fw), valid(fa2 * fw)
    var = np.maximum(s2 - s1 * s1 / tpl.sw, 1e-9)
    return num / (np.sqrt(var) * tpl.tn)


def _peak(surface: np.ndarray) -> tuple[float, float, float, bool]:
    """(dx, dy, score, at_edge) with a parabolic sub-pixel refinement."""
    r = surface.shape[0] // 2
    iy, ix = np.unravel_index(int(np.argmax(surface)), surface.shape)
    score = float(surface[iy, ix])
    at_edge = ix in (0, surface.shape[1] - 1) or iy in (0, surface.shape[0] - 1)

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else max(-0.5, min(0.5, 0.5 * (a - c) / d))

    fx = sub(surface[iy, ix - 1], score, surface[iy, ix + 1]) if not at_edge else 0.0
    fy = sub(surface[iy - 1, ix], score, surface[iy + 1, ix]) if not at_edge else 0.0
    return ix - r + fx, iy - r + fy, score, at_edge


def _ncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0:
        return 0.0
    am, bm = (a * w).sum() / sw, (b * w).sum() / sw
    az, bz = a - am, b - bm
    den = math.sqrt(float((w * az * az).sum()) * float((w * bz * bz).sum()))
    return float((w * az * bz).sum() / den) if den > 0 else 0.0


# --- scene base -----------------------------------------------------------------------------


@dataclass
class BaseResult:
    png: bytes
    mask_png: bytes
    width_px: int
    height_px: int
    print_box_px: tuple[int, int, int, int]
    protected_box_px: tuple[int, int, int, int]
    ppi: float


def _png_text(png: bytes, key: bytes, text: str) -> bytes:
    data = key + b"\x00" + text.encode("latin-1")
    chunk = struct.pack(">I", len(data)) + b"tEXt" + data
    chunk += struct.pack(">I", zlib.crc32(b"tEXt" + data) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    return png[:ihdr_end] + chunk + png[ihdr_end:]


def read_base_meta(path: Path) -> tuple[str, str] | None:
    """(garment, view) written by `render_base` into the base PNG, or None."""
    data = path.read_bytes()[:65536]
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    i = 8
    while i + 8 <= len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        kind, body = data[i + 4 : i + 8], data[i + 8 : i + 8 + n]
        if kind == b"tEXt" and body.startswith(BASE_META_KEY + b"\x00"):
            garment, _, view = body[len(BASE_META_KEY) + 1 :].decode("latin-1").partition("/")
            return garment, view
        if kind == b"IDAT":
            return None
        i += 12 + n
    return None


def _protected_box(maps: TemplateMaps) -> tuple[int, int, int, int]:
    x, y, w, h = maps.print_area_px
    m = round(MASK_MARGIN_IN * maps.ppi)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(maps.width, x + w + m), min(maps.height, y + h + m)
    return x0, y0, x1 - x0, y1 - y0


def render_base(garment: str, view: str, width: int, height: int, blank_hex: str) -> BaseResult:
    if view not in SCENE_VIEWS:
        raise ImageError(f"view must be one of {', '.join(SCENE_VIEWS)}")
    maps = template_maps(garment, view, width, height)
    rgb = tuple(parse_hex(blank_hex)[:3])
    fabric = photos._garment_rgb(maps, rgb).rint().cast("uchar")
    bg = LIFESTYLE_BG if view == "lifestyle_base" else (255, 255, 255)
    img = photos.layer(maps, fabric, fabric, bg)
    png = img.pngsave_buffer(compression=6, profile="srgb", keep=8)
    png = _png_text(png, BASE_META_KEY, f"{garment}/{view}")

    px, py, pw, ph = _protected_box(maps)
    alpha = pyvips.Image.black(width, height).draw_rect(255, px, py, pw, ph, fill=True)
    mask = pyvips.Image.black(width, height, bands=3).bandjoin(alpha).cast("uchar")
    mask_png = mask.copy(interpretation="srgb").pngsave_buffer(compression=9)
    return BaseResult(
        png, mask_png, width, height, maps.print_area_px, (px, py, pw, ph), round(maps.ppi, 4)
    )


# --- scene input ----------------------------------------------------------------------------


def load_scene(path: Path) -> pyvips.Image:
    """The provider's scene as 3-band sRGB uchar, or SceneInputError (format, size)."""
    kind = sniff_format(path)
    if kind not in SCENE_FORMATS:
        raise SceneInputError("scene must be a PNG, JPEG or WebP image")
    try:
        img = load(path, access="random")
    except ImageError as e:
        raise SceneInputError(f"scene: {e}") from e
    if err := pixel_budget_error(img.width, img.height, "scene"):
        raise SceneInputError(err)
    if not (SCENE_MIN_SIDE <= img.width <= SCENE_MAX_SIDE) or not (
        SCENE_MIN_SIDE <= img.height <= SCENE_MAX_SIDE
    ):
        raise SceneInputError(
            f"scene is {img.width} x {img.height} px; each side must be "
            f"{SCENE_MIN_SIDE}-{SCENE_MAX_SIDE} px"
        )
    try:
        if img.interpretation not in ("srgb", "b-w") or img.format != "uchar":
            img = img.colourspace("srgb").cast("uchar")
        if img.bands in (1, 2):
            img = img.colourspace("srgb")
        if img.hasalpha():
            img = img.flatten(background=[255, 255, 255])
        img = img[0:3].cast("uchar").copy(interpretation="srgb")
        return img.copy_memory()
    except pyvips.Error as e:
        raise SceneInputError(f"scene: unreadable image: {str(e).splitlines()[0]}") from e


# --- registration ---------------------------------------------------------------------------


@dataclass
class Registration:
    scale: float  # base px -> fitted scene px
    tx: float  # base px
    ty: float
    fit: float  # scene px -> fitted scene px (cover fit)
    ox: float  # cover-fit crop, fitted scene px
    oy: float
    score: float
    at_limit: bool

    def scene_to(self, k: float, e: tuple[float, float]) -> tuple[float, tuple[float, float]]:
        """(m, off) such that frame = m * scene + off, for a frame where frame = k*base + e."""
        m = k * self.fit / self.scale
        ox = -k * (self.tx + self.ox) / self.scale + e[0]
        oy = -k * (self.ty + self.oy) / self.scale + e[1]
        return m, (ox, oy)


def register(scene: pyvips.Image, base: pyvips.Image, maps: TemplateMaps) -> Registration:
    wb, hb = base.width, base.height
    fit = max(wb / scene.width, hb / scene.height)
    ox, oy = (scene.width * fit - wb) / 2, (scene.height * fit - hb) / 2
    longest = max(wb, hb)

    # Template: base edges over the product's bounding box, grown 3%.
    bx, by, bw, bh = maps.product_bbox_px
    grow = 0.03 * longest

    def template(r: float):
        w, h = round(wb * r), round(hb * r)
        eb = _edges(_warp(base, r, (0, 0), w, h))
        # Weights: the product (garment and figure), dilated by about 1% of the frame.
        prod = _warp(maps.product.cast("float") / 255, r, (0, 0), w, h).numpy()
        wt = np.clip(_gauss(prod, max(1.0, 0.01 * longest * r)) * 4, 0, 1)
        x0 = max(1, round((bx - grow) * r))
        y0 = max(1, round((by - grow) * r))
        x1 = min(eb.shape[1] - 1, round((bx + bw + grow) * r))
        y1 = min(eb.shape[0] - 1, round((by + bh + grow) * r))
        return _Template(eb[y0:y1, x0:x1], wt[y0:y1, x0:x1]), x0, y0

    def scene_edges(r: float, s: float, tx: float, ty: float) -> np.ndarray:
        reg = Registration(s, tx, ty, fit, ox, oy, 0.0, False)
        m, off = reg.scene_to(r, (0.0, 0.0))
        return _edges(_warp(scene, m, off, round(wb * r), round(hb * r)))

    rc = REG_COARSE_PX / longest
    tc, x0, y0 = template(rc)
    rad = max(2, round(REG_MAX_SHIFT * REG_COARSE_PX))
    coarse = []
    for s in REG_SCALES:
        dx, dy, score, edge = _peak(_ncc_search(scene_edges(rc, s, 0, 0), tc, x0, y0, rad))
        coarse.append((s, s * dx / rc, s * dy / rc, score, edge))
    s_c, _, _, _, edge_c = max(coarse, key=lambda c: c[3])

    # Scale and translation trade off (a figure cut by the frame's bottom edge grows downward
    # out of the frame), so the fine search covers +-2 coarse steps of scale, each starting
    # from the coarse translation interpolated at that scale.
    scales = [c[0] for c in coarse]
    rf = min(1.0, REG_FINE_PX / longest)
    tf, x0, y0 = template(rf)
    rad_f = math.ceil(rf / rc) + 2
    fine = None
    step = REG_SCALES[1] - REG_SCALES[0]
    n = round(2 * step / REG_FINE_STEP)
    for i in range(-n, n + 1):
        s = round(s_c + i * REG_FINE_STEP, 4)
        if not REG_SCALES[0] <= s <= REG_SCALES[-1]:
            continue
        tx0 = float(np.interp(s, scales, [c[1] for c in coarse]))
        ty0 = float(np.interp(s, scales, [c[2] for c in coarse]))
        e = scene_edges(rf, s, tx0, ty0)
        dx, dy, score, edge = _peak(_ncc_search(e, tf, x0, y0, rad_f))
        if fine is None or score > fine[3]:
            fine = (s, tx0 + s * dx / rf, ty0 + s * dy / rf, score, edge)
    s, tx, ty, score, edge_f = fine
    at_limit = edge_c or edge_f or s in (REG_SCALES[0], REG_SCALES[-1])
    return Registration(s, tx, ty, fit, ox, oy, score, at_limit)


# --- check 1 --------------------------------------------------------------------------------


@dataclass
class OutlineCheck:
    score: float
    gain: list[float]  # per channel: scene garment / base garment in the ring, clamped


def outline_check(
    scene: pyvips.Image, base: pyvips.Image, maps: TemplateMaps, reg: Registration
) -> OutlineCheck:
    """Is there an edge in the registered scene where the base's garment outline is?

    `near` is 1 on the outline (the coverage masks' boundaries) and falls to 0 about BAND_IN
    away. In the zone (a ring around the print box), the score is the weighted correlation of
    the scene's edge map with `near`, relative to the same correlation for the base itself
    (capped at 1). A relit scene or a new background changes edge contrast, not where the edges
    are; a moved or reshaped garment puts its edges off the outline and the score drops."""
    wb, hb = base.width, base.height
    r = min(1.0, CHECK1_PX / max(wb, hb))
    w, h = round(wb * r), round(hb * r)
    base_r = _warp(base, r, (0, 0), w, h)
    m, off = reg.scene_to(r, (0.0, 0.0))
    scene_r = _warp(scene, m, off, w, h)
    eb, es = _edges(base_r, 0.7, 0, False), _edges(scene_r, 0.7, 0, False)

    ppi = maps.ppi * r
    sigma = max(0.5, BAND_IN * ppi / 2)

    def nearness(mask: pyvips.Image) -> np.ndarray:
        soft = _warp(mask.cast("float"), r, (0, 0), w, h).gaussblur(sigma, precision="float")
        return np.clip(1 - np.abs(2 * soft.numpy() / 255 - 1), 0, 1)

    # The visible outline: where the garment meets the background or the figure, and where the
    # front of the garment meets what is behind it (a hood).
    front = _warp(maps.front.cast("float"), r, (0, 0), w, h)
    near = np.maximum(nearness(maps.front), nearness(maps.product))
    if maps.back is not None:
        near = np.maximum(near, nearness(maps.back))
    band = near > 0.02

    x, y, bw, bh = (v * r for v in maps.print_area_px)
    ring, inner = RING_IN * ppi, MASK_MARGIN_IN * ppi
    yy, xx = np.mgrid[0:h, 0:w]

    def inside(g):
        return (xx >= x - g) & (xx < x + bw + g) & (yy >= y - g) & (yy < y + bh + g)

    zone = inside(ring) & ~inside(inner)
    weight = (band & zone).astype(np.float64)
    ref = _ncc(eb, near, weight)
    score = _ncc(es, near, weight) / ref if ref > 0 else 0.0
    score = min(1.0, score)

    # Global light and white balance: mean of garment pixels in the ring, scene over base, per
    # channel. The luminance gain is clamped to GAIN_RANGE and each channel to within GAIN_TINT
    # of it, so a scene's warm light carries onto the restored blank without a visible seam,
    # but no scene can recolor it.
    body = (front.numpy() > 250) & zone & ~band
    if not body.any():
        return OutlineCheck(round(score, 4), [1.0, 1.0, 1.0])
    b, sc = base_r.numpy()[body].astype(np.float64), scene_r.numpy()[body].astype(np.float64)
    lum = np.array([0.2126, 0.7152, 0.0722])
    g = float((sc @ lum).mean() / max(1.0, (b @ lum).mean()))
    g = min(GAIN_RANGE[1], max(GAIN_RANGE[0], g))
    per = sc.mean(0) / np.maximum(4.0, b.mean(0))
    gain = [round(float(min(g * (1 + GAIN_TINT), max(g * (1 - GAIN_TINT), v))), 4) for v in per]
    return OutlineCheck(round(score, 4), gain)


def print_box_foreign(
    scene: pyvips.Image, base: pyvips.Image, maps: TemplateMaps, reg: Registration
) -> float:
    """Share of the print area where the registered scene has an edge the base's blank doesn't.

    Works on relative contrast (gradient of log luminance), so relighting, tint and smooth
    folds stay near 0 whatever the blank's color, while marks the provider drew into the box
    (text, a logo, a disc: sharp steps) stand out. The base's own edges (template seams,
    drawstrings) are dilated by a few pixels and subtracted, so a sub-pixel registration error
    on them doesn't count."""
    wb, hb = base.width, base.height
    ax, ay, aw, ah = maps.print_area_px
    r = min(1.0, BOX_PX / max(aw, ah))
    w, h = round(wb * r), round(hb * r)
    m, off = reg.scene_to(r, (0.0, 0.0))
    x0, y0 = round(ax * r), round(ay * r)
    bw, bh = max(1, round(aw * r)), max(1, round(ah * r))

    def rel_edges(img: pyvips.Image) -> np.ndarray:
        lum = _luma(img).gaussblur(BOX_BLUR_PX, precision="float") + BOX_LUMA_FLOOR
        lg = lum.log().crop(x0, y0, bw, bh).numpy().astype(np.float32)
        gy, gx = np.gradient(lg)
        return np.hypot(gx, gy)

    es = rel_edges(_warp(scene, m, off, w, h))
    eb = rel_edges(_warp(base, r, (0, 0), w, h))
    eb = pyvips.Image.new_from_array(np.ascontiguousarray(eb)).rank(5, 5, 24).numpy()
    excess = es - eb
    return round(float((excess > BOX_EDGE_MIN).mean()), 4)


# --- check 2 --------------------------------------------------------------------------------


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    img = pyvips.Image.new_from_array(np.ascontiguousarray(a, dtype=np.float32))
    return img.gaussblur(sigma, precision="float").numpy()


def _small(img: pyvips.Image) -> pyvips.Image:
    k = min(1.0, SSIM_PX / max(img.width, img.height))
    return img.resize(k, kernel="linear") if k < 1 else img


def design_lock_score(
    printed: pyvips.Image, reference: pyvips.Image, weight: pyvips.Image
) -> float:
    """SSIM (mean of R, G, B; weighted by the design's coverage) of the printed region after a
    smooth achromatic de-shade, against the reference (source design flat on the blank). All
    three images are the region's size; `weight` is 0..1."""
    p = _small(printed.cast("float")).numpy().astype(np.float64)
    q = _small(reference.cast("float")).numpy().astype(np.float64)
    wt = _small(weight.cast("float")).numpy().astype(np.float64)
    if wt.ndim == 3:
        wt = wt[..., 0]
    hh, ww = wt.shape
    sig = DESHADE_SIGMA * max(hh, ww)
    lum = np.array([0.2126, 0.7152, 0.0722])
    lp, lq = _gauss(p @ lum, sig), _gauss(q @ lum, sig)
    d = np.clip(lq / np.maximum(lp, 1.0), *DESHADE_RANGE)
    p = np.clip(p * d[..., None], 0, 255)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    maps = []
    for ch in range(3):
        x, y = p[..., ch], q[..., ch]
        mx, my = _gauss(x, SSIM_SIGMA), _gauss(y, SSIM_SIGMA)
        sxx = _gauss(x * x, SSIM_SIGMA) - mx * mx
        syy = _gauss(y * y, SSIM_SIGMA) - my * my
        sxy = _gauss(x * y, SSIM_SIGMA) - mx * my
        maps.append(
            ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        )
    ssim = np.mean(maps, axis=0)
    w = _gauss(wt, 2.0) + 0.02
    return round(float((ssim * w).sum() / w.sum()), 4)


# --- composite ------------------------------------------------------------------------------


@dataclass
class SceneResult:
    data: bytes | None
    format: str
    width_px: int
    height_px: int
    print_box_px: tuple[int, int, int, int]
    passes: bool
    failures: list[str]
    region_unchanged_score: float
    design_lock_score: float | None
    registration: dict
    upscale: float
    underbase_applied: bool
    preset_failures: list[str] = field(default_factory=list)


def _frame(preset: photos.Preset, maps_b: TemplateMaps, garment: str, view: str):
    """Template maps at the output frame (the base frame scaled to cover the preset) and the
    crop that cuts the preset out of it, keeping the print area inside."""
    W, H = preset.width, preset.height
    kk = max(W / maps_b.width, H / maps_b.height)
    wf, hf = max(W, math.ceil(maps_b.width * kk)), max(H, math.ceil(maps_b.height * kk))
    maps = template_maps(garment, view, wf, hf)
    ax, ay, aw, ah = maps.print_area_px

    def pick(center: int, lo: int, hi: int) -> int:
        return max(0, min(max(center, lo), hi))

    # Centered on the product where possible, always keeping the print area in the frame.
    bx, by, bw, bh = maps.product_bbox_px
    cx = pick(bx + (bw - W) // 2, ax + aw - W, min(wf - W, ax))
    cy = pick(by + (bh - H) // 2, ay + ah - H, min(hf - H, ay))
    k = maps.ppi / maps_b.ppi
    e = (
        (maps_b.origin_in[0] - maps.origin_in[0]) * maps.ppi,
        (maps_b.origin_in[1] - maps.origin_in[1]) * maps.ppi,
    )
    return maps, (cx, cy), k, e


def _shade_field(scene_patch, blank_patch, sigma):
    """Local light of the scene relative to the (gain-adjusted) blank, smoothed and clamped."""
    ls = _luma(scene_patch).gaussblur(sigma, precision="float")
    lb = _luma(blank_patch).gaussblur(sigma, precision="float")
    ratio = (ls + 4) / (lb + 4)
    lo, hi = SHADE_RANGE
    return (ratio < lo).ifthenelse(lo, (ratio > hi).ifthenelse(hi, ratio))


def composite_scene(
    scene_path: Path,
    base_path: Path,
    print_box_px: tuple[int, int, int, int],
    design: Path,
    w_in: float,
    h_in: float,
    blank_hex: str,
    preset_name: str,
    fmt: str,
    xmp_subjects: list[str],
    underbase_preview: bool = False,
    garment: str | None = None,
    view: str | None = None,
) -> SceneResult:
    if preset_name == "amazon_main":
        raise ImageError("a lifestyle scene can't be the Amazon main image; use amazon_alt")
    preset = photos.PRESETS[preset_name]
    scene = load_scene(scene_path)
    base = load(base_path, access="random")
    base = base.flatten(background=[255, 255, 255]) if base.hasalpha() else base
    base = base[0:3].cast("uchar").copy_memory()

    meta = read_base_meta(base_path)
    if garment is None or view is None:
        if meta is None:
            raise ImageError("base_key is not a scene base; pass garment and view")
        garment, view = garment or meta[0], view or meta[1]
    try:
        maps_b = template_maps(garment, view, base.width, base.height)
    except KeyError as e:
        raise ImageError(f"base has no template: {garment}/{view}") from e
    if view not in SCENE_VIEWS:
        raise ImageError(f"view must be one of {', '.join(SCENE_VIEWS)}")
    if any(abs(a - b) > 1 for a, b in zip(print_box_px, maps_b.print_area_px, strict=True)):
        raise ImageError(
            f"print_box_px {list(print_box_px)} doesn't match the base's print area "
            f"{list(maps_b.print_area_px)}"
        )

    reg = register(scene, base, maps_b)
    check1 = outline_check(scene, base, maps_b, reg)
    reg_info = {
        "scale": round(reg.scale, 4),
        "dx_px": round(reg.tx, 2),
        "dy_px": round(reg.ty, 2),
        "score": round(reg.score, 4),
        "at_limit": reg.at_limit,
    }
    W, H = preset.width, preset.height
    maps, (cx, cy), k, e = _frame(preset, maps_b, garment, view)
    m, off = reg.scene_to(k, (e[0] - cx, e[1] - cy))
    x, y, dw, dh, scale = photos._place(maps, w_in, h_in)
    box = (x - cx, y - cy, dw, dh)
    rgb = tuple(parse_hex(blank_hex)[:3])
    underbase = underbase_preview and photos.luma(rgb) < photos.UNDERBASE_LUMA

    failures = []
    # Check 1b: the provider drew nothing into the print area (a mask it ignored, the garment
    # outline intact): reported as region_changed too, nothing is restored over it.
    foreign = print_box_foreign(scene, base, maps_b, reg)
    if check1.score < REGION_UNCHANGED_MIN or reg.at_limit or foreign > BOX_FOREIGN_MAX:
        failures.append("region_changed")
        return SceneResult(
            None, fmt, W, H, box, False, failures, check1.score, None, reg_info, m, underbase
        )

    out_scene = _warp(scene, m, off, W, H).copy_memory()  # read twice: patch and frame
    out = _restore_and_print(
        out_scene, maps, (cx, cy), check1.gain, design, w_in, h_in, rgb, underbase
    )
    data = photos.encode(out, fmt, xmp_subjects)

    decoded = pyvips.Image.new_from_buffer(data, "")[0:3]
    lock = lock_check(decoded, maps, (cx, cy), design, w_in, h_in, rgb, underbase)
    if lock < DESIGN_LOCK_MIN:
        failures.append("design_drift")

    preset_failures = []
    if scale < 1.0:
        preset_failures.append("design_larger_than_print_area")
    if preset.white_required:
        preset_failures.append("background_not_white")
    if max(decoded.width, decoded.height) < preset.min_longest_px:
        preset_failures.append("too_small")
    return SceneResult(
        None if failures else data,
        fmt,
        W,
        H,
        box,
        not failures,
        failures,
        check1.score,
        lock,
        reg_info,
        m,
        underbase,
        preset_failures,
    )


def _restore_and_print(
    out_scene: pyvips.Image,
    maps: TemplateMaps,
    crop: tuple[int, int],
    gain: list[float],
    design: Path,
    w_in: float,
    h_in: float,
    rgb: tuple[int, int, int],
    underbase: bool,
) -> pyvips.Image:
    """Restore the blank print area (plus MASK_MARGIN_IN) into the scene, lit by the scene, and
    print the design into it. Works on that patch only; the rest of the frame is the scene."""
    cx, cy = crop
    W, H = out_scene.width, out_scene.height
    ppi = maps.ppi
    ax, ay, aw, ah = maps.print_area_px
    mg = round(MASK_MARGIN_IN * ppi)
    # The patch, in frame coordinates, clipped to the output window.
    fx0, fy0 = max(ax - mg, cx), max(ay - mg, cy)
    fx1, fy1 = min(ax + aw + mg, cx + W), min(ay + ah + mg, cy + H)
    pw, ph = fx1 - fx0, fy1 - fy0
    if pw <= 0 or ph <= 0:
        return out_scene

    def crop_map(img):
        return img.crop(fx0, fy0, pw, ph)

    scene = out_scene.crop(fx0 - cx, fy0 - cy, pw, ph).cast("float").copy_memory()
    shadow = crop_map(maps.shadow).cast("float") / 255
    hl = crop_map(maps.highlight).cast("float")
    strength = 0.3 + 0.3 * (1 - photos.luma(rgb))
    blank = 255 - (255 - shadow * list(rgb)) * (1 - hl * (strength / 255))
    blank = blank * gain
    shade = _shade_field(scene, blank, max(0.5, SHADE_SIGMA_IN * ppi)).copy_memory()
    fabric = blank * shade
    front_a = crop_map(maps.front).cast("float") / 255

    x, y, dw, dh, scale = photos._place(maps, w_in, h_in)
    px0, py0 = max(x, fx0), max(y, fy0)
    px1, py1 = min(x + dw, fx1), min(y + dh, fy1)
    printed = fabric
    if px1 > px0 and py1 > py0:
        qw, qh = px1 - px0, py1 - py0
        lx, ly = px0 - fx0, py0 - fy0
        art = photos._design(design, dw, dh, ppi * scale)
        # Displacement: the template's folds plus the scene's light read as a height map.
        relief = (shade - 1).gaussblur(max(0.5, HEIGHT_SIGMA_IN * ppi), precision="float")
        height = crop_map(maps.height_map).cast("float") + relief * FOLD_HEIGHT_PER_SHADE
        sx, sy = photos._shift(height.crop(lx, ly, qw, qh), ppi)
        lim = photos.FOLD_SHIFT_IN * ppi  # never more than the template's steepest fold
        sx = (sx > lim).ifthenelse(lim, (sx < -lim).ifthenelse(-lim, sx))
        sy = (sy > lim).ifthenelse(lim, (sy < -lim).ifthenelse(-lim, sy))
        index = pyvips.Image.xyz(qw, qh).cast("float") + [px0 - x, py0 - y]
        art = art.mapim(index - sx.bandjoin(sy)).unpremultiply()
        art_rgb, art_a = art[0:3].cast("float"), art[3].cast("float") / 255
        base = fabric.crop(lx, ly, qw, qh)
        if underbase:
            g = art_a * photos.UNDERBASE_GAIN
            u = (g > 1).ifthenelse(1.0, g)
            white = photos._garment_rgb_patch(maps, px0, py0, qw, qh) * gain
            base = base * (1 - u) + white * shade.crop(lx, ly, qw, qh) * u
        a = art_a * front_a.crop(lx, ly, qw, qh)
        light = (1 - (1 - shadow.crop(lx, ly, qw, qh)) * 0.5) * shade.crop(lx, ly, qw, qh) * gain
        ink = art_rgb * light
        patch = base * (1 - a) + ink * a
        printed = fabric.insert(patch, lx, ly)

    # Feather: 1 over the print area and most of the margin, fading to the scene at its edge,
    # and only on the garment.
    rect = pyvips.Image.black(pw, ph).draw_rect(
        255, ax - fx0 - mg // 2, ay - fy0 - mg // 2, aw + mg, ah + mg, fill=True
    )
    feather = rect.gaussblur(max(0.5, mg / 6), precision="float") / 255 * front_a
    feather = (feather > 1).ifthenelse(1.0, feather)
    mixed = scene * (1 - feather) + printed * feather
    mixed = (mixed > 255).ifthenelse(255, (mixed < 0).ifthenelse(0, mixed))
    patch8 = mixed.rint().cast("uchar").copy_memory()
    return out_scene.insert(patch8, fx0 - cx, fy0 - cy).copy(interpretation="srgb")


def lock_check(
    out: pyvips.Image,
    maps: TemplateMaps,
    crop: tuple[int, int],
    design: Path,
    w_in: float,
    h_in: float,
    rgb: tuple[int, int, int],
    underbase: bool,
) -> float:
    """Check 2 on an output frame: the printed region against `design` (the source)."""
    x, y, dw, dh, scale = photos._place(maps, w_in, h_in)
    printed = out.crop(x - crop[0], y - crop[1], dw, dh)
    reference, weight = _reference(maps, design, (x, y, dw, dh), scale, rgb, underbase)
    return design_lock_score(printed, reference, weight)


def _reference(maps, design, box, scale, rgb, underbase):
    """The source design scaled to its box, flat on the exact blank (underbase rule as in the
    composite), clipped by the garment; and its coverage as the SSIM weight."""
    x, y, dw, dh = box
    art = photos._design(design, dw, dh, maps.ppi * scale).unpremultiply()
    a = art[3].cast("float") / 255 * (maps.front.crop(x, y, dw, dh).cast("float") / 255)
    under = (pyvips.Image.black(dw, dh, bands=3) + list(rgb)).cast("float")
    if underbase:
        g = art[3].cast("float") / 255 * photos.UNDERBASE_GAIN
        u = (g > 1).ifthenelse(1.0, g)
        under = under * (1 - u) + 255 * u
    ref = under * (1 - a) + art[0:3].cast("float") * a
    return ref, a
