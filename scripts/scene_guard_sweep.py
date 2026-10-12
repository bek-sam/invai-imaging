"""Check 1 margin and the placement guard, measured offline (T-40-2, B-374).

For every row the real scene lock runs in process (register, outline check, print-box check, and
the placement guard when `app.scenes` has one), on local files, with no network and nothing
paid. Nothing here opens an image for a person: the output is a TSV of numbers.

    uv run python scripts/scene_guard_sweep.py --sweep --out /tmp/x/rows.tsv \
        [--garments tee hoodie] [--blanks white sport ...] [--views lifestyle_base]
    uv run python scripts/scene_guard_sweep.py --real /tmp/w39/trial1 /tmp/w39/trial2 --out real.tsv
    uv run python scripts/scene_guard_sweep.py --summarize /tmp/x/rows*.tsv real.tsv

Sets (column `set`):
  bad   the garment is redrawn on an unchanged backdrop and print box (the mask-ignoring
        provider of `old_mask_only`): narrowed 5/10/15%, widened 10%, moved 0.3/0.6/1.0 in
        along each axis (both directions), garment-only scale 0.9/1.1; backdrops 105, 200, 235.
  good  obedient frames, re-rendered frames, whole-frame shift and 0.92-1.08 scale (registration
        absorbs these on purpose, `REG_SCALES`/`REG_MAX_SHIFT`), provider sizes, new background,
        relit and tinted, noise, fold shading of +-15% and +-30%.
  real  every `<folder>/scenes/<id>-a<n>.png` of a trial folder (see scene_calibrate.py).
Blanks: white, light (heather), mid (gray), dark (navy), black and sport-gray. A row's verdict
is check 1 as `composite_scene` applies it: no reason in `region_reasons`.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import pyvips
from scene_helpers import (
    flat_backdrop,
    folds,
    obeys_mask,
    old_mask_only,
    perturbed,
    rerendered,
)

from app import scenes
from app.garments import SPECS, template_maps

BLANKS = {
    "white": "#F2F2F2",
    "light": "#C8CACE",
    "mid": "#808080",
    "dark": "#1F2A44",
    "black": "#111111",
    "sport": "#9AA0A6",
}
TONES = (105, 200, 235)
PASS_MIN = scenes.REGION_UNCHANGED_MIN
GUARD = getattr(scenes, "placement_check", None)

# case name -> keyword arguments of `old_mask_only`
BAD = {}
for pct in (5, 10, 15):
    BAD[f"narrow{pct}"] = {"widen": 1 - pct / 100}
BAD["widen10"] = {"widen": 1.1}
for d in (0.3, 0.6, 1.0):
    for sign in ("+", "-"):
        v = d if sign == "+" else -d
        BAD[f"move-x{sign}{d}"] = {"dx_in": v}
        BAD[f"move-y{sign}{d}"] = {"shift_in": v}
BAD["gscale0.9"] = {"scale": 0.9}
BAD["gscale1.1"] = {"scale": 1.1}

# name -> (scale, shift px at 1024, provider size)
FRAME = {
    "t-mock": (1.02, (7, -5)),
    "t-s0.92": (0.92, (0, 0)),
    "t-s1.08": (1.08, (0, 0)),
    "t-sh+x40": (1.0, (40, 0)),
    "t-sh-y40": (1.0, (0, -40)),
    "t-s0.95sh": (0.95, (-30, 30)),
    "t-s1.05sh": (1.05, (30, -30)),
}
SIZES = ((1024, 1024), (1536, 1024))

COLUMNS = (
    "set", "case", "garment", "view", "blank", "tone", "c1", "reg", "scale", "at_limit",
    "box_foreign", "w_in", "h_in", "x_in", "y_in", "reasons", "pass",
)  # fmt: skip


def measure(scene: pyvips.Image, base: pyvips.Image, maps, extra: dict) -> dict:
    scene = scene.copy_memory()  # lazy frames would be re-evaluated ~20 times by registration
    reg = scenes.register(scene, base, maps)
    c1 = scenes.outline_check(scene, base, maps, reg)
    foreign = scenes.print_box_foreign(scene, base, maps, reg)
    row = dict(extra)
    row.update(c1=f"{c1.score:.4f}", reg=f"{reg.score:.4f}", scale=f"{reg.scale:.4f}")
    row.update(at_limit=str(reg.at_limit).lower(), box_foreign=f"{foreign:.4f}")
    pl = GUARD(scene, base, maps, reg) if GUARD else None
    kw = {"placement": pl} if pl is not None else {}
    reasons = scenes.region_reasons(c1.score, reg.at_limit, foreign, **kw)
    for k, attr in (("w_in", "width_in"), ("h_in", "height_in"), ("x_in", "shift_x_in"),
                    ("y_in", "shift_y_in")):  # fmt: skip
        row[k] = f"{getattr(pl, attr):.3f}" if pl is not None else ""
    row["reasons"] = ",".join(reasons) or "-"
    row["pass"] = "yes" if not reasons else "NO"
    return row


def write(out, row: dict) -> None:
    out.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")
    out.flush()


def sweep(args) -> int:
    out = args.out.open("w") if args.out else sys.stdout
    out.write("\t".join(COLUMNS) + "\n")
    size = args.size
    for g in args.garments:
        for v in args.views:
            maps = template_maps(g, v, size, size)
            for bname in args.blanks:
                hexv = BLANKS[bname]
                res = scenes.render_base(g, v, size, size, hexv)
                base = pyvips.Image.new_from_buffer(res.png, "")[0:3].copy_memory()
                pb = res.protected_box_px

                def row(set_, case, tone, scene, _g=g, _v=v, _b=bname, _base=base, _m=maps):
                    ex = {
                        "set": set_,
                        "case": case,
                        "garment": _g,
                        "view": _v,
                        "blank": _b,
                        "tone": tone,
                    }
                    write(out, measure(scene, _base, _m, ex))

                for tone in TONES:
                    gray = flat_backdrop(size, size, tone)
                    obedient = obeys_mask(res.png, res.mask_png, 0.92, gray)
                    row("good", "a", tone, obedient)
                    row("good", "a-rerendered", tone, rerendered(obedient))
                    if tone == TONES[-1]:
                        for depth in (0.15, 0.30):
                            row(
                                "good",
                                f"folds{depth}",
                                tone,
                                folds(obedient, maps.product_bbox_px, depth),
                            )
                    for case, kw in BAD.items():
                        scene = old_mask_only(res.png, pb, g, v, hexv, background=gray, **kw)
                        row("bad", case, tone, scene)
                for name, (sc, sh) in FRAME.items():
                    for sz in SIZES:
                        scene = perturbed(base, g, v, sz, scale=sc, shift=sh)
                        row("good", f"{name}@{sz[0]}x{sz[1]}", "room", scene)
    if args.out:
        out.close()
    return 0


def real(args) -> int:
    out = args.out.open("w") if args.out else sys.stdout
    out.write("\t".join(COLUMNS) + "\n")
    for folder in args.real:
        for line in (folder / "compositions.jsonl").read_text().splitlines():
            comp = json.loads(line)
            if not comp.get("scene_print_box_px"):
                continue
            base_path = folder / "scenes" / f"{comp['id']}-base.png"
            base = pyvips.Image.new_from_file(str(base_path))[0:3].copy_memory()
            g, v = scenes.read_base_meta(base_path)
            maps = template_maps(g, v, base.width, base.height)
            for attempt in (1, 2):
                sp = folder / "scenes" / f"{comp['id']}-a{attempt}.png"
                if not sp.exists():
                    continue
                ex = {
                    "set": "real",
                    "case": f"{folder.name}:{comp['id'][:8]}#{attempt}",
                    "garment": g,
                    "view": comp.get("scene_kind") or "",
                    "blank": comp["color_hex"],
                    "tone": "",
                }
                write(out, measure(scenes.load_scene(sp), base, maps, ex))
    if args.out:
        out.close()
    return 0


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        lines = p.read_text().splitlines()
        cols = lines[0].split("\t")
        rows += [dict(zip(cols, ln.split("\t"), strict=False)) for ln in lines[1:]]
    for r in rows:
        r["c1"] = float(r["c1"])
        for k in ("w_in", "h_in", "x_in", "y_in"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
        # The thresholds below are read from `scenes` now, so a re-summary after moving them
        # needs no new sweep: the other reasons are as recorded, `placement_off` is recomputed.
        reasons = [x for x in r["reasons"].split(",") if x not in ("-", "placement_off")]
        r["pass_without_guard"] = "yes" if not reasons else "NO"
        if r["w_in"] is not None and (
            max(abs(r["w_in"]), abs(r["h_in"])) > scenes.PLACEMENT_SIZE_MAX_IN
            or max(abs(r["x_in"]), abs(r["y_in"])) > scenes.PLACEMENT_SHIFT_MAX_IN
        ):
            reasons.append("placement_off")
        r["pass"] = "yes" if not reasons else "NO"
    return rows


def summarize(paths: list[Path]) -> int:
    rows = load_rows(paths)
    real_ = [r for r in rows if r["set"] == "real"]
    good = [r for r in rows if r["set"] == "good"]
    bad = [r for r in rows if r["set"] == "bad"]
    print(f"{len(rows)} rows: {len(real_)} real, {len(good)} good, {len(bad)} known-bad")

    # Check 1 as it stands: the score alone, and the verdict (no reason).
    real_pass = [r for r in real_ if r["pass"] == "yes"]
    print(f"\nREAL FRAMES: {len(real_pass)} of {len(real_)} pass check 1 (verdict)")
    if real_pass:
        lo = min(real_pass, key=lambda r: r["c1"])
        print(
            f"  lowest real-pass c1 {lo['c1']:.3f} ({lo['case']}); gap above {PASS_MIN}: {lo['c1'] - PASS_MIN:.3f}"
        )
    gl = min(good, key=lambda r: r["c1"]) if good else None
    if gl:
        print(f"GOOD (synthetic): lowest c1 {gl['c1']:.3f} ({gl['case']} {gl['garment']} {gl['blank']} tone {gl['tone']}); "
              f"{sum(r['pass'] == 'NO' for r in good)} of {len(good)} fail check 1 on contrast alone")  # fmt: skip
    if bad:
        hi = max(bad, key=lambda r: r["c1"])
        n_c1 = sum(r["c1"] >= PASS_MIN for r in bad)
        n_ver = sum(r["pass"] == "yes" for r in bad)
        print(f"KNOWN-BAD: highest c1 {hi['c1']:.3f} ({hi['case']} {hi['garment']} {hi['blank']} tone {hi['tone']}); "
              f"gap to {PASS_MIN}: {PASS_MIN - hi['c1']:.3f}")  # fmt: skip
        print(
            f"  rows with c1 >= {PASS_MIN}: {n_c1} of {len(bad)}; rows passing check 1 (verdict): {n_ver} of {len(bad)}"
        )
        print(
            "\n  class          n   max c1  rows c1>=0.75  rows pass  | width/height/shift (max |.|)"
        )
        for case in BAD:
            sel = [r for r in bad if r["case"] == case]
            if not sel:
                continue
            m = max(r["c1"] for r in sel)
            tail = ""
            if sel[0]["w_in"] is not None:
                tail = "  | " + " ".join(
                    f"{k}:{min(abs(r[k]) for r in sel):.2f}-{max(abs(r[k]) for r in sel):.2f}"
                    for k in ("w_in", "h_in", "x_in", "y_in")
                )
            print(f"  {case:<12} {len(sel):>3}   {m:.3f}   {sum(r['c1'] >= PASS_MIN for r in sel):>8}   "
                  f"{sum(r['pass'] == 'yes' for r in sel):>8}{tail}")  # fmt: skip
    if any(r.get("w_in") is not None for r in rows):
        guard_report(real_, good, bad)
    return 0


def guard_report(real_: list[dict], good: list[dict], bad: list[dict]) -> None:
    """Per measure: the highest known-good value, the lowest value of each reshaped class, both
    gaps to the threshold in inches and as a share of it (target >= 25%, ADR 0039), and the
    known-bad rows that still pass (named by class, with what they are)."""
    known_good = [r for r in real_ if r["pass_without_guard"] == "yes"] + good

    def size(r):
        return max(abs(r["w_in"]), abs(r["h_in"]))

    def shift(r):
        return max(abs(r["x_in"]), abs(r["y_in"]))

    reshaped = [c for c in BAD if c.startswith(("narrow", "widen"))]
    moved = [c for c in BAD if not c.startswith(("narrow", "widen"))]
    print(
        "\nGUARD (inches at the template ppi; size = max(|width|, |height|), shift = max(|x|, |y|))"
    )
    for name, fn, thr, classes in (
        ("size", size, scenes.PLACEMENT_SIZE_MAX_IN, reshaped),
        ("shift", shift, scenes.PLACEMENT_SHIFT_MAX_IN, moved),
    ):
        gmax = max(known_good, key=fn)
        g_c1 = [r for r in known_good if r["pass_without_guard"] == "yes"]
        rp = [r for r in real_ if r["pass_without_guard"] == "yes"]
        print(f"  {name} threshold {thr} in")
        print(f"    known-good ({len(known_good)} frames) highest {fn(gmax):.3f} "
              f"({gmax['set']} {gmax['case']} {gmax['garment']} {gmax['blank']}): gap_good "
              f"{thr - fn(gmax):.3f} in = {100 * (thr - fn(gmax)) / thr:.0f}% of threshold")  # fmt: skip
        print(f"    of those passing check 1 ({len(g_c1)}) highest {max(map(fn, g_c1), default=0):.3f}; "
              f"real passes ({len(rp)}) highest {max(map(fn, rp), default=0):.3f}")  # fmt: skip
        for c in classes:
            v = sorted(fn(r) for r in bad if r["case"] == c)
            if not v:
                continue
            n_over = sum(x > thr for x in v)
            n_room = sum(x >= 1.25 * thr for x in v)
            print(f"    {c:<11} min {v[0]:.3f} median {v[len(v) // 2]:.3f}: gap_bad(min) "
                  f"{v[0] - thr:+.3f} in = {100 * (v[0] - thr) / thr:+.0f}%; over threshold "
                  f"{n_over}/{len(v)}, with 25% room {n_room}/{len(v)}")  # fmt: skip
    print("\nVERDICT, known-bad rows failing check 1 (without guard -> with guard), by class:")
    tot0 = tot1 = 0
    for c in BAD:
        sel = [r for r in bad if r["case"] == c]
        if not sel:
            continue
        f0 = sum(r["pass_without_guard"] == "NO" for r in sel)
        f1 = sum(r["pass"] == "NO" for r in sel)
        tot0, tot1 = tot0 + f0, tot1 + f1
        print(f"  {c:<12} {f0:>3} -> {f1:>3} of {len(sel)}")
    n = len(bad)
    print(
        f"  all rows     {tot0:>3} -> {tot1:>3} of {n}; passing check 1: {n - tot0} -> {n - tot1}"
    )
    passing = [r for r in bad if r["pass"] == "yes"]
    by_class = {}
    for r in passing:
        by_class[r["case"]] = by_class.get(r["case"], 0) + 1
    print("  still passing (rigid move or scale of the garment alone: registration absorbs it, "
          "the print follows the garment): " + ", ".join(f"{k} {v}" for k, v in by_class.items()))  # fmt: skip
    gf = [r for r in known_good if r["pass"] == "NO"]
    gf0 = [r for r in known_good if r["pass_without_guard"] == "NO"]
    print(f"KNOWN-GOOD frames failing check 1: {len(gf0)} -> {len(gf)} of {len(known_good)} "
          f"(new failures caused by the guard: {len(gf) - len(gf0)})")  # fmt: skip
    print(f"REAL frames passing check 1: {sum(r['pass_without_guard'] == 'yes' for r in real_)} -> "
          f"{sum(r['pass'] == 'yes' for r in real_)} of {len(real_)}")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--real", nargs="+", type=Path)
    ap.add_argument("--summarize", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--garments", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--blanks", nargs="+", default=list(BLANKS), choices=list(BLANKS))
    ap.add_argument("--views", nargs="+", default=["lifestyle_base"])
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if args.real:
        return real(args)
    if args.sweep:
        return sweep(args)
    ap.error("one of --sweep, --real, --summarize")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
