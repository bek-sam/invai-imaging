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
    ap.add_argument("--tone", type=float, default=1.0, help="light of the drawn room (0.5 = dark)")
    ap.add_argument("--garments", nargs="*", default=list(SPECS))
    args = ap.parse_args()
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
        (r["garment"], r["view"], r["blank"]): r["pass"] == "yes"
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
        if not ok and case in ("a", "a'") and not oracle[(r["garment"], r["view"], r["blank"])]:
            print("   ^ fails like the oracle: check 1 itself rejects this garment in this room")
            r["_known"] = True
            continue
        if not ok:
            bad += 1
            print(
                f"   ^ UNEXPECTED (expected pass={want}) reasons={r['_reasons']} {r['_failures']}"
            )
    known = sum(1 for r in table if r.get("_known"))
    if known:
        print(f"\n{known} obedient-provider rows fail only where the oracle fails too")
    print(f"\n{len(table)} cases, {bad} unexpected")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
