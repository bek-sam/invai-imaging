"""Why does check 1 fail on a real scene? Offline diagnosis of provider frames (T-39-1).

    uv run python scripts/scene_outline_diag.py FOLDER [FOLDER ...]     # real frames
    uv run python scripts/scene_outline_diag.py --synthetic             # fix the verdict thresholds

A folder is a trial copy: `scenes/<id>-{base,mask,a1,a2}.png`, optionally `compositions.jsonl`
(scene kind, blank color) and `design-*.png`. Nothing is written and no network is used. For each
frame the scene lock itself is run (`composite_scene`: registration, check 1, check 1b, check 2
when check 1 passes) and then three numbers that say *why* check 1 scored what it did:

  fidelity   after registration and the per-channel light gain check 1 computes, the frame is
             compared with the base inside the mask's keep region (alpha 255) eroded by BAND_IN:
             `mad` = mean absolute luma difference (0-255), `ssim` = mean luma SSIM. "The provider
             kept our garment" is measured, not assumed.
  contrast   `step_base` / `step_frame`: median luma step (0-255) across the garment outline in
             the check-1 zone, garment side minus background side, base and frame; and
             `edge_base` / `edge_frame`: the strongest luma gradient (levels/px, sigma 0.7, what
             check 1 correlates) at the outline, so a thin drawn outline stroke counts.
  spill      share of the 0.35 in band just outside the outline whose pixels are within
             SPILL_TOL levels of the blank color (after the gain): a garment widened into the
             transparent area, which fidelity can't see. Also reports `backdrop`, the median luma
             (imaging's scale, 0-1) of the frame outside the keep region grown by 1 in.

Verdicts (thresholds below, fixed from the synthetic distributions; `--synthetic` reprints them).
Fidelity and contrast values were fixed before real frames were scored. THREE spill/backdrop
values were moved AFTER seeing the real frames, each toward "undecidable": SPILL_MAX_DARK 0.03 ->
0.50, BACKDROP_APART 0.08 -> 0.15, and "redrawn" now needs spill >= SPILL_REDRAWN 0.25 (it was
any spill above SPILL_MAX 0.03). On the 6 failing studio frames of wave 38 the original values
give 3 redrawn (both black-tee frames and white d6ea2019#2), 1 ours low contrast, 2 undecidable;
the current ones give 0 redrawn, 1 ours low contrast, 5 undecidable (re-run in round 2):
  redrawn             fidelity poor (mad > FID_MAD_BAD or ssim < FID_SSIM_BAD), or the garment
                      spilled into a backdrop that is clearly not the blank's color.
  ours, low contrast  fidelity good, no spill, and the outline edge in the frame is below
                      CONTRAST_LOW_RATIO of the base's (the garment is ours, the edge is faint).
  undecidable         anything else (fidelity between good and bad, a faint edge with spill, an
                      outline that is simply gone); it stays a failing frame.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import numpy as np
import pyvips

from app import scenes
from app.fonts import parse_hex
from app.garments import template_maps
from app.vips import load

# --- verdict thresholds (T-39-1; see the --synthetic table in the README paragraph) ----------
# Garment fidelity after the affine fit. Synthetic distributions (python scripts/... --synthetic):
# obedient re-rendered frames (noise, blur, JPEG q85) mad 1.9-2.0 / ssim 0.91-0.93; an unrelated
# garment under the same box mad 3.1-14.7 (5.7-6.1 on white, 11-15 on black except on near-black
# backdrops) / ssim 0.77-0.98. Good = inside the first, bad = clearly in the second; between is
# undecidable, never "ours".
FID_MAD_GOOD = 2.6  # levels (0-255)
FID_SSIM_GOOD = 0.89
FID_MAD_BAD = 5.0
FID_SSIM_BAD = 0.80
SPILL_TOL = 12  # per channel, levels: "looks like the blank" for the spill share
SPILL_MAX = 0.03  # share of the outer band that looks like the blank
# On a near-black blank the studio's dark contact shadow and vignette read as "blank" (real black
# tee frames: 0.04-0.15 with the garment kept), so only a band mostly filled counts (a widened
# black tee: 0.87-0.97). Changed after the first scoring of the real frames; see the report.
SPILL_REDRAWN = 0.25  # between SPILL_MAX and this the garment edge may have grown a little
SPILL_MAX_DARK = 0.50
DARK_BLANK = 0.15  # blank luma (0-1) below this uses SPILL_MAX_DARK
BACKDROP_APART = (
    0.15  # |backdrop luma (gain-adjusted) - blank luma| (0-1) above this: spill can be told apart
)
CONTRAST_LOW_RATIO = 0.60  # frame outline edge / base outline edge below this: a faint edge
CONTRAST_GONE_RATIO = 0.15  # below this the outline is not there at all (a widened or absent
# garment measures <= 0.11 on the synthetic set; obedient frames never go under 0.35)

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _disc_mask(r: int) -> pyvips.Image:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disc = np.where(xx * xx + yy * yy <= r * r, 255, 128).astype(np.uint8)  # 128 = ignore
    return pyvips.Image.new_from_memory(disc.tobytes(), 2 * r + 1, 2 * r + 1, 1, "uchar")


def morph(mask: np.ndarray, r: int, op: str) -> np.ndarray:
    """Binary erode/dilate by a true disc of radius r px."""
    if r <= 0:
        return mask
    img = pyvips.Image.new_from_array((mask * 255).astype(np.uint8))
    return img.morph(_disc_mask(r), op).numpy() > 127


def _erode1(b: np.ndarray) -> np.ndarray:
    out = b.copy()
    out[1:] &= b[:-1]
    out[:-1] &= b[1:]
    out[:, 1:] &= b[:, :-1]
    out[:, :-1] &= b[:, 1:]
    return out


def luma255(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = scenes._gauss
    ma, mb = g(a, sigma), g(b, sigma)
    saa, sbb, sab = g(a * a, sigma) - ma * ma, g(b * b, sigma) - mb * mb, g(a * b, sigma) - ma * mb
    return ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma * ma + mb * mb + c1) * (saa + sbb + c2))


def zone_mask(maps, w: int, h: int) -> np.ndarray:
    """The ring check 1 looks at: RING_IN around the print area, minus MASK_MARGIN_IN."""
    x, y, bw, bh = maps.print_area_px
    ring, inner = scenes.RING_IN * maps.ppi, scenes.MASK_MARGIN_IN * maps.ppi
    yy, xx = np.mgrid[0:h, 0:w]

    def inside(g):
        return (xx >= x - g) & (xx < x + bw + g) & (yy >= y - g) & (yy < y + bh + g)

    return inside(ring) & ~inside(inner)


def outline_step(lum: np.ndarray, product: np.ndarray, zone: np.ndarray, ppi: float) -> float:
    """Median |luma step| across the garment outline: mean luma of the garment side minus the
    background side within 0.15 in, sampled on boundary pixels inside the zone."""
    e = morph(product, 3, "erode")
    d = ~morph(product, 3, "dilate")
    s = max(2.0, 0.15 * ppi)
    g = scenes._gauss
    ge, gd = g(e.astype(np.float32), s), g(d.astype(np.float32), s)
    lg = g((lum * e).astype(np.float32), s) / np.maximum(ge, 1e-3)
    lb = g((lum * d).astype(np.float32), s) / np.maximum(gd, 1e-3)
    bnd = product & ~_erode1(product) & zone & (ge > 0.05) & (gd > 0.05)
    if not bnd.any():
        return 0.0
    return float(np.median(np.abs(lg - lb)[bnd]))


def edge_peak(lum: np.ndarray, product: np.ndarray, zone: np.ndarray) -> float:
    """Median over the outline pixels (in the zone) of the strongest luma gradient (levels/px,
    sigma 0.7, the edge map check 1 correlates) within 2 px: how sharp the outline really is."""
    g = scenes._gauss(lum.astype(np.float32), 0.7)
    gy, gx = np.gradient(g)
    gm = np.hypot(gx, gy)
    pad = np.pad(gm, 2, mode="edge")
    pk = np.zeros_like(gm)
    for dy in range(5):
        for dx in range(5):
            pk = np.maximum(pk, pad[dy : dy + gm.shape[0], dx : dx + gm.shape[1]])
    bnd = product & ~_erode1(product) & zone
    return float(np.median(pk[bnd])) if bnd.any() else 0.0


def c1_parts(scene, base, maps, reg) -> tuple[float, float]:
    """Check 1's score (same formula, same zone, 1 px = 1 px) restricted to the garment's own
    outline (`maps.front`) and to the rest of the product outline (the figure's neck, arms,
    legs): which of the two the failure comes from. nan when the part has no pixels in the zone."""
    w, h = base.width, base.height
    m, off = reg.scene_to(1.0, (0.0, 0.0))
    eb = scenes._edges(base, 0.7, 0, False)
    es = scenes._edges(scenes._warp(scene, m, off, w, h), 0.7, 0, False)
    sigma = max(0.5, scenes.BAND_IN * maps.ppi / 2)

    def nearness(mask: pyvips.Image) -> np.ndarray:
        soft = mask.cast("float").gaussblur(sigma, precision="float").numpy()
        return np.clip(1 - np.abs(2 * soft / 255 - 1), 0, 1)

    zone = zone_mask(maps, w, h)
    n_front, n_prod = nearness(maps.front), nearness(maps.product)
    n_back = nearness(maps.back) if maps.back is not None else np.zeros_like(n_front)
    n_gar = np.maximum(n_front, n_back)
    near = np.maximum(n_gar, n_prod)
    parts = []
    for sel in ((n_gar > 0.02), (n_prod > 0.02) & ~(n_gar > 0.02)):
        wt = (sel & zone).astype(np.float64)
        if wt.sum() < 50:
            parts.append(float("nan"))
            continue
        ref = scenes._ncc(eb, near, wt)
        parts.append(min(1.0, scenes._ncc(es, near, wt) / ref) if ref > 0 else 0.0)
    return parts[0], parts[1]


def measure(
    scene: pyvips.Image,
    base: pyvips.Image,
    maps,
    keep: np.ndarray,
    blank_rgb: tuple[int, int, int],
) -> dict:
    """Registration, check 1 and the three diagnosis numbers for one frame."""
    reg = scenes.register(scene, base, maps)
    chk = scenes.outline_check(scene, base, maps, reg)
    c1_gar, c1_fig = c1_parts(scene, base, maps, reg)
    w, h = base.width, base.height
    m, off = reg.scene_to(1.0, (0.0, 0.0))
    frame = scenes._warp(scene, m, off, w, h).numpy().astype(np.float64)
    base_a = base.numpy().astype(np.float64)
    gain = np.array(chk.gain)
    adj = np.clip(frame / gain, 0, 255)
    ppi = maps.ppi
    r_band = round(scenes.BAND_IN * ppi)

    # Fidelity is judged on the garment (maps.front) and, separately, on the figure behind it.
    # The gain check 1 computes is a property of the garment, so the figure is compared raw
    # (applying the gain to a mannequin that the provider left alone reads as 80 levels of
    # change). Provider "black" is often lifted by an additive offset a gain can't express, so
    # after the gain a luma affine fit (slope, offset) is removed from the garment before MAD
    # and SSIM; what remains is structure: folds, seams, outline.
    front_m = maps.front.numpy() > 127
    prod_m = maps.product.numpy() > 127
    lb, la = luma255(base_a), luma255(adj)
    inner = morph(keep & front_m, r_band, "erode")
    fig_in = morph(keep & prod_m & ~front_m, r_band, "erode")
    a_fit, b_fit = np.polyfit(lb[inner], la[inner], 1) if inner.any() else (1.0, 0.0)
    fitted = (a_fit * lb + b_fit).astype(np.float32)
    mad = float(np.abs(la - fitted)[inner].mean()) if inner.any() else float("nan")
    sim = float(ssim(la.astype(np.float32), fitted)[inner].mean()) if inner.any() else float("nan")
    lf_raw = luma255(frame)
    mad_fig = float(np.abs(lf_raw - lb)[fig_in].mean()) if fig_in.any() else float("nan")

    product = maps.product.numpy() > 127
    zone = zone_mask(maps, w, h)
    step_base = outline_step(lb, product, zone, ppi)
    step_frame = outline_step(luma255(frame), product, zone, ppi)
    edge_base = edge_peak(lb, product, zone)
    edge_frame = edge_peak(luma255(frame), product, zone)

    band = ~product & morph(product, r_band, "dilate") & zone
    blank = np.array(blank_rgb, dtype=np.float64)
    near_blank = np.abs(adj - blank).max(axis=-1) <= SPILL_TOL
    spill = float(near_blank[band].mean()) if band.any() else 0.0

    far = ~morph(keep, round(1.0 * ppi), "dilate")
    backdrop = float(np.median(luma255(frame)[far]) / 255) if far.any() else float("nan")
    backdrop_adj = float(np.median(la[far]) / 255) if far.any() else float("nan")
    return {
        "reg": reg.score,
        "at_limit": reg.at_limit,
        "c1": chk.score,
        "c1_gar": c1_gar,
        "c1_fig": c1_fig,
        "mad": mad,
        "ssim": sim,
        "mad_fig": mad_fig,
        "fit_a": float(a_fit),
        "step_base": step_base,
        "step_frame": step_frame,
        "edge_base": edge_base,
        "edge_frame": edge_frame,
        "spill": spill,
        "backdrop": backdrop,
        "backdrop_adj": backdrop_adj,
    }


def verdict(r: dict, blank_luma: float) -> str:
    if r["at_limit"] or r["reg"] < scenes.STRUCTURAL_REG_MAX:
        return "redrawn"  # no scale + shift lines the frame up: the garment was re-posed
    if r["mad"] > FID_MAD_BAD or r["ssim"] < FID_SSIM_BAD:
        return "redrawn"
    if r["mad"] > FID_MAD_GOOD or r["ssim"] < FID_SSIM_GOOD:
        return "undecidable"
    dark = blank_luma < DARK_BLANK
    if r["spill"] > (SPILL_MAX_DARK if dark else SPILL_MAX):
        apart = abs(r["backdrop_adj"] - blank_luma) > BACKDROP_APART
        if apart and r["spill"] >= (SPILL_MAX_DARK if dark else SPILL_REDRAWN):
            return "redrawn"  # the garment grew into the transparent area
        return "undecidable"
    ratio = r["edge_frame"] / r["edge_base"] if r["edge_base"] > 0 else 1.0
    if CONTRAST_GONE_RATIO <= ratio < CONTRAST_LOW_RATIO:
        return "ours, low contrast"
    return "undecidable"


# --- real frames ------------------------------------------------------------------------------


def frames(folder: Path):
    """(id, try, scene, base, mask, scene_kind, blank_hex) for every frame in a trial folder."""
    comps = {}
    jl = folder / "compositions.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            c = json.loads(line)
            comps[c["id"]] = c
    for base in sorted((folder / "scenes").glob("*-base.png")):
        cid = base.name[: -len("-base.png")]
        c = comps.get(cid, {})
        for a in (1, 2):
            sp = folder / "scenes" / f"{cid}-a{a}.png"
            if sp.exists():
                yield (
                    cid,
                    a,
                    sp,
                    base,
                    folder / "scenes" / f"{cid}-mask.png",
                    c.get("scene_kind") or "control",
                    c.get("color_hex") or "#000000",
                )


def diagnose(folder: Path, design: Path, w_in: float, h_in: float) -> list[dict]:
    out = []
    for cid, a, sp, bp, mp, kind, hexv in frames(folder):
        garment, view = scenes.read_base_meta(bp)
        base = load(bp, access="random")
        base = base[0:3].cast("uchar").copy_memory()
        maps = template_maps(garment, view, base.width, base.height)
        res_box = scenes.render_base(garment, view, base.width, base.height, hexv).print_box_px
        res = scenes.composite_scene(
            sp, bp, res_box, design, w_in, h_in, hexv, "etsy", "jpeg", [], underbase_preview=True
        )
        keep = pyvips.Image.new_from_file(str(mp))[3].numpy() == 255
        rgb = tuple(parse_hex(hexv)[:3])
        m = measure(scenes.load_scene(sp), base, maps, keep, rgb)
        blank_luma = (np.array(rgb) @ LUMA) / 255
        out.append(
            {
                "folder": folder.name,
                "id": cid[:8],
                "try": a,
                "kind": kind,
                "blank": hexv,
                "pass": "yes" if res.passes else "NO",
                "lock": "" if res.design_lock_score is None else f"{res.design_lock_score:.3f}",
                "verdict": "" if res.passes else verdict(m, blank_luma),
                **m,
                "c1_run": res.region_unchanged_score,
            }
        )
    return out


COLS = (
    ("folder", "folder"),
    ("id", "id"),
    ("try", "try"),
    ("kind", "kind"),
    ("blank", "blank"),
    ("reg", "reg"),
    ("c1", "c1"),
    ("c1_gar", "c1gar"),
    ("c1_fig", "c1fig"),
    ("lock", "lock"),
    ("mad", "madG"),
    ("ssim", "ssimG"),
    ("mad_fig", "madF"),
    ("step_base", "stepB"),
    ("step_frame", "stepF"),
    ("edge_base", "edgeB"),
    ("edge_frame", "edgeF"),
    ("spill", "spill"),
    ("backdrop", "backdrop"),
    ("backdrop_adj", "bdAdj"),
    ("pass", "pass"),
    ("verdict", "verdict"),
)


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_table(rows: list[dict]) -> None:
    cells = [[fmt(r[k]) for k, _ in COLS] for r in rows]
    heads = [h for _, h in COLS]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(heads)]
    print("  ".join(h.ljust(w) for h, w in zip(heads, widths, strict=True)))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths, strict=True)))


# --- synthetic distributions (fix the thresholds) -----------------------------------------------

SYN_BLANKS = {"white": "#F2F2F2", "black": "#111111"}
SYN_TONES = (250, 235, 215, 190, 160, 128, 96, 64, 32, 10)


def synthetic(size: int = 1024) -> None:
    from scene_helpers import (
        flat_backdrop,
        moved_garment,
        obeys_mask,
        old_mask_only,
        rerendered,
    )

    rows = []
    g, v = "tee", "lifestyle_base"
    other = "hoodie"
    for blank, hexv in SYN_BLANKS.items():
        res = scenes.render_base(g, v, size, size, hexv)
        base_img = pyvips.Image.new_from_buffer(res.png, "")[0:3].copy_memory()
        maps = template_maps(g, v, size, size)
        keep = pyvips.Image.new_from_buffer(res.mask_png, "")[3].numpy() == 255
        rgb = tuple(parse_hex(hexv)[:3])
        ores = scenes.render_base(other, v, size, size, hexv)
        oimg = pyvips.Image.new_from_buffer(ores.png, "")[0:3]
        omaps = template_maps(other, v, size, size)
        for tone in SYN_TONES:
            gray = flat_backdrop(size, size, tone)
            cases = {
                "ours": rerendered(obeys_mask(res.png, res.mask_png, 0.92, gray)),
                "widen10": old_mask_only(
                    res.png, res.protected_box_px, g, v, hexv, widen=1.1, background=gray
                ),
                "narrow10": old_mask_only(
                    res.png, res.protected_box_px, g, v, hexv, widen=0.9, background=gray
                ),
                "shift0.6": old_mask_only(
                    res.png, res.protected_box_px, g, v, hexv, shift_in=0.6, background=gray
                ),
                "other-garment": omaps.product.ifthenelse(oimg.cast("float"), gray, blend=True)
                .cast("uchar")
                .copy(interpretation="srgb"),
            }
            if tone == round(float(np.array(rgb) @ LUMA)):
                pass
            for name, scene in cases.items():
                m = measure(scene, base_img, maps, keep, rgb)
                rows.append({"cls": name, "blank": blank, "tone": tone, **m})
        # a widened garment on a backdrop of exactly the blank's luma
        lvl = float(np.array(rgb) @ LUMA)
        gray = flat_backdrop(size, size, lvl)
        wide = old_mask_only(res.png, res.protected_box_px, g, v, hexv, widen=1.1, background=gray)
        rows.append(
            {"cls": "widen10@blank", "blank": blank, "tone": round(lvl),
             **measure(wide, base_img, maps, keep, rgb)}
        )  # fmt: skip
        _ = moved_garment
    classes = sorted({r["cls"] for r in rows})
    print(
        "class          n   ssim min/max     mad min/max     spill min/max   c1 min/max   edgeF/edgeB"
    )
    for c in classes:
        rs = [r for r in rows if r["cls"] == c]

        def mm(k, rs=rs):
            vals = [r[k] for r in rs]
            return f"{min(vals):.3f}/{max(vals):.3f}"

        ratio = [r["edge_frame"] / r["edge_base"] for r in rs if r["edge_base"] > 0]
        print(
            f"{c:<13} {len(rs):>3}  {mm('ssim'):<15} {mm('mad'):<14} {mm('spill'):<14} "
            f"{mm('c1'):<12} {min(ratio):.2f}/{max(ratio):.2f}"
        )
    print("\nper row (cls blank tone): c1 ssim mad spill stepB stepF")
    for r in rows:
        print(
            f"{r['cls']:<13} {r['blank']:<5} {r['tone']:>3}  c1 {r['c1']:.3f} ssim {r['ssim']:.3f} "
            f"mad {r['mad']:.2f} spill {r['spill']:.3f} stepB {r['step_base']:.1f} "
            f"stepF {r['step_frame']:.1f} edgeB {r['edge_base']:.1f} edgeF {r['edge_frame']:.1f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", type=Path, nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--design-in", type=float, nargs=2, default=(10.5, 12.0), metavar=("W", "H"))
    args = ap.parse_args()
    if args.synthetic:
        synthetic()
        return 0
    rows = []
    for f in args.folders:
        design = args.design or next(f.glob("design-*.png"))
        rows += diagnose(f, design, *args.design_in)
    print_table(rows)
    failing = [r for r in rows if r["pass"] == "NO"]
    print(f"\n{len(rows)} frames, {len(failing)} fail")
    for r in failing:
        print(
            f"  {r['folder']}/{r['id']}#{r['try']} {r['kind']}: {r['verdict']} "
            f"(ssim {r['ssim']:.3f}, mad {r['mad']:.2f}, spill {r['spill']:.3f}, "
            f"step {r['step_frame']:.1f} vs base {r['step_base']:.1f}, "
            f"edge {r['edge_frame']:.1f} vs base {r['edge_base']:.1f}, "
            f"backdrop {r['backdrop']:.2f}, c1 {r['c1']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
