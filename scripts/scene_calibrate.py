"""Offline replay of real provider scenes through the scene lock (T-37-1).

    uv run python scripts/scene_calibrate.py /tmp/invai-w37-frames

Reads `<folder>/compositions.jsonl` and `<folder>/scenes/<id>-{a1,a2,base}.png` (plus
`<folder>/design-*.png`, the design the scenes were made for, 10.5 x 12 in unless --design-in is
given) and prints one row per frame: registration, check 1, check 1b, the reasons, whether the
failure is structural, check 2 (when check 1 passes) and the verdict. Nothing is written to
storage and no network is used; images in the folder are never modified.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scenes

COLUMNS = (
    "composition",
    "try",
    "scene_kind",
    "c1",
    "at_limit",
    "scale",
    "dx",
    "dy",
    "reg",
    "box_foreign",
    "reasons",
    "structural",
    "lock",
    "pass",
)


def rows(folder: Path, design: Path, w_in: float, h_in: float) -> list[dict]:
    out = []
    for line in (folder / "compositions.jsonl").read_text().splitlines():
        comp = json.loads(line)
        if not comp.get("scene_print_box_px"):
            continue
        cid = comp["id"]
        base = folder / "scenes" / f"{cid}-base.png"
        for attempt in (1, 2):
            scene = folder / "scenes" / f"{cid}-a{attempt}.png"
            if not scene.exists():
                continue
            res = scenes.composite_scene(
                scene,
                base,
                tuple(comp["scene_print_box_px"]),
                design,
                w_in,
                h_in,
                comp["color_hex"],
                "etsy",
                "jpeg",
                [],
                underbase_preview=True,
                # The base PNG records its own garment and view (what the backend relies on).
            )
            reg = res.registration
            out.append(
                {
                    "composition": cid[:8],
                    "try": attempt,
                    "scene_kind": comp.get("scene_kind") or "",
                    "c1": f"{res.region_unchanged_score:.3f}",
                    "at_limit": str(reg["at_limit"]).lower(),
                    "scale": f"{reg['scale']:.3f}",
                    "dx": f"{reg['dx_px']:.1f}",
                    "dy": f"{reg['dy_px']:.1f}",
                    "reg": f"{reg['score']:.3f}",
                    "box_foreign": "" if res.box_foreign is None else f"{res.box_foreign:.4f}",
                    "reasons": ",".join(res.region_reasons) or "-",
                    "structural": str(res.structural).lower(),
                    "lock": "" if res.design_lock_score is None else f"{res.design_lock_score:.3f}",
                    "pass": "yes" if res.passes else "NO",
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--design-in", type=float, nargs=2, default=(10.5, 12.0), metavar=("W", "H"))
    args = ap.parse_args()
    design = args.design or next(args.folder.glob("design-*.png"))
    table = rows(args.folder, design, *args.design_in)
    widths = [max(len(c), *(len(str(r[c])) for r in table)) for c in COLUMNS]
    print("  ".join(c.ljust(w) for c, w in zip(COLUMNS, widths, strict=True)))
    for r in table:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(COLUMNS, widths, strict=True)))
    n_pass = sum(r["pass"] == "yes" for r in table)
    print(f"\n{n_pass}/{len(table)} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
