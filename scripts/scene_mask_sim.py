"""Offline proof that the scene edit mask, not the checks, is the lever (T-38-1).

    uv run python scripts/scene_mask_sim.py [--size 1024] [--frames /tmp/invai-w37-frames] \
        [--work /tmp/invai-w38-T-38-1/sim]

For every garment x view x blank (white, black) it builds the base and mask with `render_base`,
makes synthetic provider frames and runs the real `composite_scene` on each (in process, local
files, no network, nothing paid):
  a   obeys the mask: keeps every opaque pixel, draws a busy scene elsewhere, light gain 0.92
  a'  obeys it but re-renders the whole frame (noise, blur, JPEG q85) at 1024x1024, 1536x1024,
      1024x1536
  a0  oracle: the base garment soft-blended (exact coverage) over the same room, light gain 0.92,
      no mask involved; what the best conceivable provider returns. Shows what check 1 itself
      allows for this garment and room.
  b   obeys only the old mask (print box + 0.5 in): the rest of the garment shifted 0.6 in or
      widened 10%
  c   re-draws the garment entirely: every kept real frame (when --frames is given), else an
      unrelated garment's base under the same box
  d   marks drawn in the print area on an otherwise obedient frame
Expected: a, a' pass; b, c fail check 1; d fails check 1b. Prints one row per case.

Backdrop-tone sweep (T-39-1): the studio scene asks the provider for a "seamless light gray
backdrop", and check 1 needs the garment's outline to stand out from it. `--sweep` replaces the
busy room with a seamless flat gray of each level in SWEEP_TONES, for the obedient (a) and
re-rendered (a') providers, on tee and hoodie, in each of SWEEP_BLANKS, and records check 1 only
(register + outline, no composite). `--bad` adds the known-bad frames on the same backdrops
(widened 10%, narrowed 10%, moved 0.6 in, another garment, garment absent, and widened on a
backdrop of exactly the blank's luma). Rows go to `--out FILE` (TSV); `--summarize FILE...`
prints the backdrop table per blank-luma band (tone with the highest worst-case check-1 margin,
c1 per tone, share at or above 0.75) and the known-bad maxima. Tone values are sRGB gray levels;
their luma on imaging's scale (photos.luma) is level / 255.

    uv run python scripts/scene_mask_sim.py --sweep [--bad] --out /tmp/x/rows.tsv \
        [--blanks white black] [--garments tee hoodie] [--views lifestyle_base]
    uv run python scripts/scene_mask_sim.py --summarize /tmp/x/rows*.tsv
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import pyvips
from PIL import Image, ImageDraw
from scene_helpers import (
    busy_scene,
    drawn_marks,
    obeys_mask,
    old_mask_only,
    rerendered,
    to_provider_size,
)

from app import scenes
from app.garments import SCENE_VIEWS, SPECS, template_maps

BLANKS = {"white": "#F2F2F2", "black": "#111111"}
COLUMNS = ("case", "garment", "view", "blank", "size", "c1", "reg", "box_foreign", "lock", "pass")
EXPECT = {
    "a0": None,
    "a": True,
    "a'": True,
    "b": False,
    "b-shift": None,
    "c": False,
    "c-real": False,
    "d": False,
}


# Backdrop sweep (T-39-1). Blank luma bands (imaging's luma scale, 0-1): the lower edge of each.
SWEEP_BLANKS = {
    "white": "#F2F2F2",
    "heather": "#C8CACE",
    "silver": "#B8B8B8",
    "sport": "#9AA0A6",
    "gray": "#808080",
    "red": "#C41E3A",
    "royal": "#2B4BA0",
    "charcoal": "#5A5F66",
    "navy": "#1F2A44",
    "forest": "#1F3D2B",
    "black": "#111111",
}
BANDS = (("white", 0.85, 1.01), ("light", 0.55, 0.85), ("mid", 0.22, 0.55), ("dark", 0.10, 0.22),
         ("black", 0.0, 0.10))  # fmt: skip
SWEEP_TONES = (250, 235, 215, 200, 190, 160, 128, 96, 64, 32, 10)  # 200 = "light gray" as drawn
LIGHT_GRAY = (
    200  # the tone the studio prompt gets today: real w38 frames' backdrop median 0.73-0.84
)
PASS_MIN = 0.75  # scenes.REGION_UNCHANGED_MIN
SAFE_MIN = 0.85  # a tone is "safe" when its worst case clears the line by this much (0.10)
EDGE_COST = 0.06  # an obedient frame may fail by this much against the oracle (max measured 0.054)
BAD_CLASSES = ("widen10", "narrow10", "shift0.6", "other-garment", "absent", "widen10@blank")
TSV_COLS = ("case", "garment", "view", "blank", "blank_luma", "tone", "c1", "reg")


def blank_luma(hexv: str) -> float:
    r, g, b = scenes.parse_hex(hexv)[:3]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def band_of(luma: float) -> str:
    return next(n for n, lo, hi in BANDS if lo <= luma < hi)


def c1_of(scene: pyvips.Image, base_img: pyvips.Image, maps) -> tuple[float, float]:
    scene = scene.copy_memory()  # the frames are lazy graphs; registration reads them ~20 times
    reg = scenes.register(scene, base_img, maps)
    return scenes.outline_check(scene, base_img, maps, reg).score, reg.score


def sweep(args) -> int:
    from scene_helpers import flat_backdrop

    out = args.out.open("w") if args.out else sys.stdout
    out.write("\t".join(TSV_COLS) + "\n")
    blanks = {k: v for k, v in SWEEP_BLANKS.items() if not args.blanks or k in args.blanks}
    size = args.size
    for g in args.garments:
        other = next(o for o in SPECS if o != g)
        for v in args.views:
            maps = template_maps(g, v, size, size)
            omaps = template_maps(other, v, size, size)
            for bname, hexv in blanks.items():
                res = scenes.render_base(g, v, size, size, hexv)
                base_img = pyvips.Image.new_from_buffer(res.png, "")[0:3].copy_memory()
                bl = blank_luma(hexv)
                ores = scenes.render_base(other, v, size, size, hexv)
                oimg = pyvips.Image.new_from_buffer(ores.png, "")[0:3]

                def row(case, tone, scene, _g=g, _v=v, _b=bname, _bl=bl, _bi=base_img, _m=maps):
                    c1, reg = c1_of(scene, _bi, _m)
                    out.write(f"{case}\t{_g}\t{_v}\t{_b}\t{_bl:.3f}\t{tone}\t{c1:.4f}\t{reg:.4f}\n")
                    out.flush()

                tones = list(SWEEP_TONES)
                for tone in tones:
                    gray = flat_backdrop(size, size, tone)
                    obedient = obeys_mask(res.png, res.mask_png, 0.92, gray)
                    row("a", tone, obedient)
                    row("a'", tone, rerendered(obedient))
                    if not args.bad:
                        continue
                    pb = res.protected_box_px
                    row(
                        "widen10",
                        tone,
                        old_mask_only(res.png, pb, g, v, hexv, widen=1.1, background=gray),
                    )
                    row(
                        "narrow10",
                        tone,
                        old_mask_only(res.png, pb, g, v, hexv, widen=0.9, background=gray),
                    )
                    row(
                        "shift0.6",
                        tone,
                        old_mask_only(res.png, pb, g, v, hexv, shift_in=0.6, background=gray),
                    )
                    oc = omaps.product.ifthenelse(oimg.cast("float"), gray, blend=True)
                    row("other-garment", tone, oc.cast("uchar").copy(interpretation="srgb"))
                    row("absent", tone, gray.cast("uchar").copy(interpretation="srgb"))
                if args.bad:
                    lvl = round(bl * 255)
                    gray = flat_backdrop(size, size, lvl)
                    pb = res.protected_box_px
                    row(
                        "widen10@blank",
                        lvl,
                        old_mask_only(res.png, pb, g, v, hexv, widen=1.1, background=gray),
                    )
    if args.out:
        out.close()
    return 0


def summarize(paths: list[Path]) -> int:
    rows = []
    for p in paths:
        lines = p.read_text().splitlines()
        for ln in lines[1:]:
            c = dict(zip(TSV_COLS, ln.split("\t"), strict=True))
            c["blank_luma"], c["tone"], c["c1"], c["reg"] = (
                float(c["blank_luma"]),
                int(c["tone"]),
                float(c["c1"]),
                float(c["reg"]),
            )
            rows.append(c)  # fmt: skip
    good = [r for r in rows if r["case"] in ("a", "a'")]
    print(f"{len(rows)} rows, {len(good)} obedient/re-rendered")
    tones = sorted({r["tone"] for r in good}, reverse=True)
    print(
        "\nBACKDROP TABLE (c1 = check 1 on a seamless flat gray; min / mean over garments x "
        "blanks in the band x {obedient, re-rendered}; share = cases >= 0.75)"
    )
    for name, lo, hi in BANDS:
        sel = [r for r in good if lo <= r["blank_luma"] < hi]
        if not sel:
            continue
        blanks = sorted({(r["blank"], r["blank_luma"]) for r in sel}, key=lambda x: -x[1])
        print(f"\nband {name}: blank luma [{lo:.2f}, {min(hi, 1.0):.2f}); blanks "
              + ", ".join(f"{b} {lm:.3f}" for b, lm in blanks))  # fmt: skip
        stats = {}
        for t in tones:
            vals = [r["c1"] for r in sel if r["tone"] == t]
            if not vals:
                continue
            stats[t] = (
                min(vals),
                sum(vals) / len(vals),
                sum(v >= PASS_MIN for v in vals) / len(vals),
            )
            print(
                f"  tone {t:>3} (luma {t / 255:.3f})  min {stats[t][0]:.3f}  mean {stats[t][1]:.3f}  share {stats[t][2]:.2f}"
            )
        best = max(stats, key=lambda t: (stats[t][0], stats[t][1]))
        ok = [t for t in stats if stats[t][2] == 1.0]
        print(f"  best tone {best} (luma {best / 255:.3f}), worst-case margin "
              f"{stats[best][0] - PASS_MIN:+.3f}; every case passes at: {ok or 'none'}")  # fmt: skip
        windows, cur = [], []
        for t in sorted(stats, reverse=True):  # light to dark
            if stats[t][0] >= SAFE_MIN:
                cur.append(t)
            elif cur:
                windows.append(cur)
                cur = []
        if cur:
            windows.append(cur)
        if windows:
            widest = max(windows, key=lambda w: w[0] - w[-1])
            mid = round((widest[0] + widest[-1]) / 2)
            print(f"  safe windows (worst case >= {SAFE_MIN:.2f}): "
                  + ", ".join(f"{w[-1]}-{w[0]}" for w in windows)
                  + f"; widest {widest[-1]}-{widest[0]}, middle {mid} (luma {mid / 255:.3f})")  # fmt: skip
        avoid = [t for t in stats if stats[t][0] < PASS_MIN]
        print(f"  fails (some case < {PASS_MIN}) at: {sorted(avoid, reverse=True) or 'none'}")
        light = stats.get(LIGHT_GRAY)
        if light:
            print(f"  today's light gray ({LIGHT_GRAY}): min {light[0]:.3f}, share {light[2]:.2f}")
    bad = [r for r in rows if r["case"] in BAD_CLASSES]
    if bad:
        print("\nKNOWN-BAD FRAMES (must score < 0.75): max c1 per class, rows >= 0.75")
        for c in BAD_CLASSES:
            sel = [r for r in bad if r["case"] == c]
            if not sel:
                continue
            top = max(sel, key=lambda r: r["c1"])
            n_pass = sum(r["c1"] >= PASS_MIN for r in sel)
            print(f"  {c:<14} n {len(sel):>3}  max c1 {top['c1']:.3f} ({top['garment']} "
                  f"{top['blank']} tone {top['tone']})  rows >= 0.75: {n_pass}")  # fmt: skip
        asserted = [r for r in bad if r["case"] != "shift0.6"]
        hi = max(asserted, key=lambda r: r["c1"])
        print(
            f"  highest asserted bad score {hi['c1']:.3f} -> gap to 0.75 is {PASS_MIN - hi['c1']:.3f}"
        )
    return 0


def make_design(path: Path) -> None:
    im = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 560, 560), fill=(230, 60, 40, 255))
    d.rectangle((200, 150, 400, 450), fill=(250, 240, 60, 255))
    d.rectangle((260, 470, 340, 500), fill=(20, 20, 120, 255))
    im.save(path)


def run(case, garment, view, blank, size, scene: pyvips.Image, base_path, box, design, work):
    sp = work / f"{case.replace("'", 'p')}-{garment}-{view}-{blank}-{size}.png"
    scene.pngsave(str(sp))
    res = scenes.composite_scene(
        sp,
        base_path,
        box,
        design,
        8,
        8,
        BLANKS[blank],
        "shopify",
        "jpeg",
        [],
        garment=garment,
        view=view,
    )
    return {
        "case": case,
        "garment": garment,
        "view": view,
        "blank": blank,
        "size": size,
        "c1": f"{res.region_unchanged_score:.3f}",
        "reg": f"{res.registration['score']:.3f}",
        "box_foreign": "" if res.box_foreign is None else f"{res.box_foreign:.4f}",
        "lock": "" if res.design_lock_score is None else f"{res.design_lock_score:.3f}",
        "pass": "yes" if res.passes else "NO",
        "_reasons": res.region_reasons,
        "_failures": res.failures,
        "_wrote": res.data is not None,
    }


def real_frames(folder: Path):
    import json

    for line in (folder / "compositions.jsonl").read_text().splitlines():
        comp = json.loads(line)
        if not comp.get("scene_print_box_px"):
            continue
        cid = comp["id"]
        for a in (1, 2):
            sp = folder / "scenes" / f"{cid}-a{a}.png"
            if sp.exists():
                yield cid[:8], a, sp, folder / "scenes" / f"{cid}-base.png", comp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--margin", type=float, default=None, help="override MASK_SILHOUETTE_MARGIN_IN")
    ap.add_argument("--tone", type=float, default=1.0, help="light of the drawn room (0.5 = dark)")
    ap.add_argument("--garments", nargs="*", default=list(SPECS))
    ap.add_argument("--sweep", action="store_true", help="backdrop-tone sweep (see above)")
    ap.add_argument("--bad", action="store_true", help="with --sweep: add the known-bad frames")
    ap.add_argument("--out", type=Path, default=None, help="with --sweep: TSV rows file")
    ap.add_argument("--blanks", nargs="*", default=None, help="with --sweep: SWEEP_BLANKS names")
    ap.add_argument("--views", nargs="*", default=["lifestyle_base"], choices=list(SCENE_VIEWS))
    ap.add_argument("--summarize", type=Path, nargs="+", default=None, metavar="TSV")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if args.sweep:
        if "--garments" not in sys.argv:
            args.garments = ["tee", "hoodie"]
        return sweep(args)
    if args.margin is not None:
        scenes.MASK_SILHOUETTE_MARGIN_IN = args.margin
    work = args.work or Path(tempfile.mkdtemp(prefix="invai-scene-sim-"))
    work.mkdir(parents=True, exist_ok=True)
    design = work / "design.png"
    make_design(design)

    table = []
    for g in args.garments:
        for v in SCENE_VIEWS:
            for blank, hexv in BLANKS.items():
                res = scenes.render_base(g, v, args.size, args.size, hexv)
                base_path = work / f"base-{g}-{v}-{blank}.png"
                base_path.write_bytes(res.png)
                box = res.print_box_px

                def go(case, scene, size="same", _g=g, _v=v, _b=blank, _bp=base_path, _x=box):
                    table.append(run(case, _g, _v, _b, size, scene, _bp, _x, design, work))

                room = busy_scene(args.size, args.size, tone=args.tone)
                maps = template_maps(g, v, args.size, args.size)
                base_img = pyvips.Image.new_from_buffer(res.png, "")[0:3]
                oracle = maps.product.ifthenelse(base_img.cast("float") * 0.92, room, blend=True)
                go("a0", oracle.cast("uchar").copy(interpretation="srgb"))
                go("a", obeys_mask(res.png, res.mask_png, 0.92, room))
                for sz in ((1024, 1024), (1536, 1024), (1024, 1536)):
                    frame = rerendered(obeys_mask(res.png, res.mask_png, 0.92, room))
                    go("a'", to_provider_size(frame, sz), f"{sz[0]}x{sz[1]}")
                go(
                    "b-shift",
                    old_mask_only(res.png, res.protected_box_px, g, v, hexv, shift_in=0.6),
                    "shift0.6",
                )
                go(
                    "b",
                    old_mask_only(res.png, res.protected_box_px, g, v, hexv, widen=1.1),
                    "widen10%",
                )
                # c: an unrelated garment's frame (synthetic stand-in for a re-drawn garment)
                other = next(o for o in SPECS if o != g)
                ores = scenes.render_base(other, v, args.size, args.size, hexv)
                oimg = pyvips.Image.new_from_buffer(ores.png, "")[0:3]
                go("c", oimg.copy(interpretation="srgb"), f"{other}-frame")
                # d: marks drawn in the print area of an obedient frame
                ink = (40, 40, 40) if blank == "white" else (230, 230, 230)
                marked = drawn_marks(obeys_mask(res.png, res.mask_png, 0.92, room), box, ink)
                go("d", marked)
    if args.frames:
        for cid, a, sp, bp, comp in real_frames(args.frames):
            d = next(args.frames.glob("design-*.png"))
            r = scenes.composite_scene(
                sp, bp, tuple(comp["scene_print_box_px"]), d, 10.5, 12.0,
                comp["color_hex"], "etsy", "jpeg", [],
            )  # fmt: skip
            table.append(
                {
                    "case": "c-real",
                    "garment": "tee",
                    "view": comp.get("scene_kind") or "",
                    "blank": f"{cid}#{a}",
                    "size": "real",
                    "c1": f"{r.region_unchanged_score:.3f}",
                    "reg": f"{r.registration['score']:.3f}",
                    "box_foreign": "" if r.box_foreign is None else f"{r.box_foreign:.4f}",
                    "lock": "" if r.design_lock_score is None else f"{r.design_lock_score:.3f}",
                    "pass": "yes" if r.passes else "NO",
                    "_reasons": r.region_reasons,
                    "_failures": r.failures,
                    "_wrote": r.data is not None,
                }
            )

    widths = [max(len(c), *(len(str(r[c])) for r in table)) for c in COLUMNS]
    print("  ".join(c.ljust(w) for c, w in zip(COLUMNS, widths, strict=True)))
    bad = 0
    oracle = {
        (r["garment"], r["view"], r["blank"]): (r["pass"] == "yes", float(r["c1"]))
        for r in table
        if r["case"] == "a0"
    }
    for r in table:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(COLUMNS, widths, strict=True)))
        case = r["case"]
        want = EXPECT[case]
        if want is None:
            continue
        ok = (r["pass"] == "yes") == want
        if case == "d":
            ok = ok and "box_foreign" in r["_reasons"] and not r["_wrote"]
        if not ok and case in ("a", "a'"):
            o_pass, o_c1 = oracle[(r["garment"], r["view"], r["blank"])]
            if not o_pass:
                print(
                    "   ^ fails like the oracle: check 1 itself rejects this garment in this room"
                )
                r["_known"] = True
                continue
            # The oracle passes, so the mask's hard edge is what costs the last points (measured
            # 0.04-0.09 on white garments on the white studio view): known, not unexpected, as long
            # as the only failing reason is the outline and the cost stays within EDGE_COST.
            if r["_reasons"] == ["outline_low"] and o_c1 - float(r["c1"]) <= EDGE_COST:
                print(f"   ^ within {EDGE_COST} of the oracle ({o_c1:.3f}): the mask's hard edge")
                r["_known"] = True
                continue
        if not ok:
            bad += 1
            print(
                f"   ^ UNEXPECTED (expected pass={want}) reasons={r['_reasons']} {r['_failures']}"
            )
    known = sum(1 for r in table if r.get("_known"))
    if known:
        print(
            f"\n{known} obedient-provider rows fail only where the oracle fails or within the "
            "mask-edge cost of it"
        )
    print(f"\n{len(table)} cases, {bad} unexpected")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
