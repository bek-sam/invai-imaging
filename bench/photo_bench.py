"""Listing-photo budget run (T-26-2). Renders with the seed's sample art and prints timings.

    uv run python bench/photo_bench.py [contact_sheet.jpg]
    /usr/bin/time -l uv run python bench/photo_bench.py   # macOS: peak RSS

- render: one `amazon_alt` (2000 x 2000 px) render per template, cold (template drawn) and warm
  (template cached), averaged;
- zip: 48 rendered JPEGs zipped the way `/photo/zip` does;
- with a path: a contact sheet of all 16 views on a black and a white blank.
"""

import sys
import tempfile
import time
from pathlib import Path

import pyvips

from app import garments, photos, sample_art


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="invai-photo-bench-"))
    art = tmp / "art.png"
    sample_art.render_sample_art("Desert Bloom", 3000, 3600, "#E4572E").save(art, dpi=(300, 300))

    cold, warm, outs = [], [], []
    for g in garments.GARMENTS:
        for v in garments.VIEWS:
            for i, hexc in enumerate(("#111111", "#F5F5F5", "#1B3A6B")):
                t = time.perf_counter()
                r = photos.render_photo(art, 10, 12, g, v, hexc, True, "amazon_alt", "jpeg", [])
                (cold if i == 0 else warm).append(time.perf_counter() - t)
                path = tmp / f"{g}-{v}-{i}.jpg"
                path.write_bytes(r.data)
                outs.append(path)
    print(f"render 2000px: cold {sum(cold) / len(cold):.3f}s avg, max {max(cold):.3f}s")
    print(f"render 2000px: warm {sum(warm) / len(warm):.3f}s avg, max {max(warm):.3f}s")

    t = time.perf_counter()
    size = photos.build_zip([(p, f"amazon/{p.name}") for p in outs[:48]], tmp / "set.zip")
    print(f"zip 48 images: {time.perf_counter() - t:.3f}s, {size / 1e6:.1f} MB")

    if len(sys.argv) > 1:
        tiles = []
        for hexc in ("#111111", "#FFFFFF"):
            for g in garments.GARMENTS:
                for v in garments.VIEWS:
                    r = photos.render_photo(art, 10, 12, g, v, hexc, True, "amazon_alt", "jpeg", [])
                    tile = pyvips.Image.new_from_buffer(r.data, "").thumbnail_image(480)
                    tiles.append(tile.draw_rect([200, 200, 200], 0, 0, 480, 480))
        sheet = pyvips.Image.arrayjoin(tiles, across=8, shim=4, background=[120, 120, 120])
        sheet.jpegsave(sys.argv[1], Q=88)
        print(f"contact sheet: {sys.argv[1]}")


if __name__ == "__main__":
    main()
