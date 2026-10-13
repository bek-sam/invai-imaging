"""Print-file checks (T-42-4, spec print-file-checks criteria 1-11, 24): thin lines, small text,
white box, color profile. Fixtures are synthetic and generated here; nothing is committed.

Strokes are drawn at 300 DPI (a 4 x 4 in design is 1200 px), so a 6 px stroke is exactly 0.020 in
and a 9 px stroke exactly 0.030 in."""

import json
import struct
import subprocess
import sys

import pytest
import pyvips
from PIL import Image, ImageCms, ImageDraw

from app import qa_print
from app.qa import check

DPI = 300


def px(inches: float) -> int:
    return round(inches * DPI)


def canvas(size_px: int = 1200) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def save(im: Image.Image, tmp_path, name="a.png", **kw):
    p = tmp_path / name
    im.save(p, **kw)
    return p


def issue(res, code):
    found = [i for i in res.issues if i.code == code]
    assert len(found) <= 1, "one issue per code per call"
    return found[0] if found else None


def line_art(stroke_px: int, size_px: int = 1200):
    """A thick body at the top and a horizontal hairline of `stroke_px` rows below it."""
    im, d = canvas(size_px)
    d.rectangle((100, 100, 700, 500), fill=(200, 30, 30, 255))
    d.rectangle((100, 800, 1100, 800 + stroke_px - 1), fill=(20, 20, 20, 255))
    return im


# --- thin_line: criteria 1-4 ------------------------------------------------------------------


def test_thin_line_error_with_measure_and_region(tmp_path):
    p = save(line_art(5), tmp_path)  # 5 px at 300 DPI = 0.0167 in
    res = check(p, 4, 4)
    i = issue(res, "thin_line")
    assert i.severity == "error"
    assert i.measured_in == pytest.approx(5 / DPI, abs=1 / DPI)  # within one analysis pixel
    assert i.regions
    r = i.regions[0]  # the box sits on the hairline (rows 800-804 of 1200), not on the body
    assert r["top"] == pytest.approx(800 / 1200, abs=0.01)
    assert r["height"] < 0.05
    assert r["left"] == pytest.approx(100 / 1200, abs=0.01)
    assert r["width"] == pytest.approx(1000 / 1200, abs=0.02)


@pytest.mark.parametrize(
    ("stroke", "expected"),
    [(4, "error"), (5, "error"), (6, "warn"), (7, "warn"), (8, "warn"), (9, None), (10, None)],
)
def test_thin_line_boundaries_at_exactly_0_020_and_0_030_in(tmp_path, stroke, expected):
    # 6 px = 0.020 in is the first warning; 9 px = 0.030 in is no issue (exact at 300 DPI).
    res = check(save(line_art(stroke), tmp_path), 4, 4)
    i = issue(res, "thin_line")
    assert (i.severity if i else None) == expected


def test_thin_line_warn_at_0_025(tmp_path):
    res = check(save(line_art(8), tmp_path), 4, 4)  # 0.0267 in
    assert [(i.code, i.severity) for i in res.issues if i.code == "thin_line"] == [
        ("thin_line", "warn")
    ]


def test_same_pixels_different_print_size_different_verdict(tmp_path):
    im, d = canvas(600)
    d.rectangle((50, 50, 350, 250), fill=(0, 0, 0, 255))
    d.rectangle((50, 400, 550, 402), fill=(0, 0, 0, 255))  # 3 px
    p = save(im, tmp_path)
    assert issue(check(p, 2, 2), "thin_line").severity == "error"  # 300 DPI: 0.010 in
    assert issue(check(p, 4, 4), "thin_line").severity == "warn"  # 150 DPI: 0.020 in
    assert issue(check(p, 8, 8), "thin_line") is None  # 75 DPI: 0.040 in


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_hairline_survives_the_analysis_downscale(tmp_path, offset):
    # 600 DPI file: analysed at 300 DPI, so a 0.0133 in line (8 px) is 4 analysis pixels, and a
    # 0.0067 in line straddling two analysis rows must still be found.
    for stroke, severity in ((8, "error"), (3, "error"), (20, None)):
        im = Image.new("RGBA", (2400, 2400), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((100, 100, 1200, 1000), fill=(0, 0, 0, 255))
        d.rectangle((100, 1500 + offset, 2200, 1500 + offset + stroke - 1), fill=(0, 0, 0, 255))
        i = issue(check(save(im, tmp_path, f"d{stroke}{offset}.png"), 4, 4), "thin_line")
        assert (i.severity if i else None) == severity


def test_no_target_size_no_size_checks(tmp_path):
    p = save(line_art(3), tmp_path)
    res = check(p)
    assert issue(res, "thin_line") is None and issue(res, "small_text") is None


def test_tapered_tips_of_thick_shapes_are_not_thin_lines(tmp_path):
    import math

    im, d = canvas(1600)
    # a 5-point star (36 degree points), a triangle, and a swash that tapers to a point
    star = []
    for k in range(10):
        r = 380 if k % 2 == 0 else 150
        a = -math.pi / 2 + k * math.pi / 5
        star.append((420 + r * math.cos(a), 420 + r * math.sin(a)))
    d.polygon(star, fill=(0, 0, 0, 255))
    d.polygon([(900, 800), (1500, 800), (1200, 100)], fill=(0, 0, 0, 255))
    d.polygon([(100, 1200), (1500, 1100), (1500, 1180), (100, 1250)], fill=(0, 0, 0, 255))
    d.polygon([(100, 1250), (1500, 1180), (1500, 1500), (100, 1500)], fill=(0, 0, 0, 255))
    d.ellipse((900, 900, 1500, 1050), outline=(0, 0, 0, 255), width=40)  # a badge ring edge
    res = check(save(im, tmp_path), 6, 6)  # 266 DPI; points are 0.02-0.03 in only at their tip
    assert issue(res, "thin_line") is None


def test_hairline_attached_to_a_star_still_flags(tmp_path):
    im, d = canvas(1200)
    d.polygon([(300, 100), (500, 600), (100, 600)], fill=(0, 0, 0, 255))
    d.rectangle((300, 700, 1100, 704), fill=(0, 0, 0, 255))  # 5 px hairline, 0.0167 in
    res = check(save(im, tmp_path), 4, 4)
    assert issue(res, "thin_line").severity == "error"


def test_diagonal_thin_line(tmp_path):
    im, d = canvas(1200)
    d.line((100, 1000, 1000, 100), fill=(0, 0, 0, 255), width=5)
    i = issue(check(save(im, tmp_path), 4, 4), "thin_line")
    assert i is not None and i.severity == "error"
    assert i.measured_in < qa_print.THIN_LINE_ERROR_IN * 1.15


def test_regions_merged_largest_first_and_capped(tmp_path):
    im, d = canvas(1200)
    for k in range(25):  # 25 separate hairlines
        y = 20 + k * 44
        d.rectangle((100, y, 100 + 100 + k * 30, y + 4), fill=(0, 0, 0, 255))
    d.rectangle((100, 1150, 1100, 1154), fill=(0, 0, 0, 255))  # the longest
    i = issue(check(save(im, tmp_path), 4, 4), "thin_line")
    assert 1 < len(i.regions) <= 20
    areas = [r["width"] * r["height"] for r in i.regions]
    assert areas == sorted(areas, reverse=True)
    assert all(0 <= v <= 1 for r in i.regions for v in r.values())


# --- small_text: criterion 5 ------------------------------------------------------------------


def glyph_row(n: int, height_px: int, size_px: int = 1200):
    im, d = canvas(size_px)
    w, gap, x = max(6, height_px // 2), max(4, height_px // 3), 100
    for _ in range(n):
        d.rectangle((x, 500, x + w, 500 + height_px), fill=(0, 0, 0, 255))
        x += w + gap
    return im


def test_small_text_row_warns(tmp_path):
    res = check(save(glyph_row(5, px(0.08)), tmp_path), 4, 4)
    i = issue(res, "small_text")
    assert i.severity == "warn" and i.regions
    assert i.measured_in == pytest.approx(0.08, abs=0.005)
    assert i.regions[0]["top"] == pytest.approx(500 / 1200, abs=0.01)


def test_taller_text_row_is_fine(tmp_path):
    assert issue(check(save(glyph_row(5, px(0.20)), tmp_path), 4, 4), "small_text") is None


def test_two_dots_are_not_a_row(tmp_path):
    assert issue(check(save(glyph_row(2, px(0.08)), tmp_path), 4, 4), "small_text") is None
    assert issue(check(save(glyph_row(3, px(0.08)), tmp_path), 4, 4), "small_text") is None


def test_small_text_is_never_an_error(tmp_path):
    res = check(save(glyph_row(8, px(0.03)), tmp_path), 4, 4)
    assert all(i.severity != "error" for i in res.issues if i.code == "small_text")


# --- white_box: criteria 6-7 ------------------------------------------------------------------


def framed(frame_rgb, size_px: int = 1800, ring: int = 40):
    im = Image.new("RGBA", (size_px, size_px), frame_rgb + (255,))
    ImageDraw.Draw(im).rectangle(
        (ring, ring, size_px - 1 - ring, size_px - 1 - ring), fill=(0, 0, 0, 0)
    )
    ImageDraw.Draw(im).ellipse((500, 500, 1300, 1300), fill=(200, 30, 30, 255))
    return im


def test_white_frame_warns_with_regions_on_the_frame(tmp_path):
    i = issue(check(save(framed((255, 255, 255)), tmp_path), 6, 6), "white_box")
    assert i.severity == "warn" and i.measured_in is None
    assert any(r["top"] == 0 and r["width"] == 1.0 for r in i.regions)  # the top strip


def test_near_white_tolerance(tmp_path):
    assert issue(check(save(framed((246, 246, 246)), tmp_path), 6, 6), "white_box")
    assert issue(check(save(framed((245, 245, 245)), tmp_path), 6, 6), "white_box")
    assert issue(check(save(framed((240, 240, 240)), tmp_path), 6, 6), "white_box") is None
    assert issue(check(save(framed((255, 255, 244)), tmp_path), 6, 6), "white_box") is None


def test_white_art_in_a_transparent_frame_is_fine(tmp_path):
    im = Image.new("RGBA", (1800, 1800), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((300, 300, 1500, 1500), fill=(255, 255, 255, 255))
    assert issue(check(save(im, tmp_path), 6, 6), "white_box") is None


def test_opaque_white_file_is_only_no_alpha(tmp_path):
    p = save(Image.new("RGB", (1200, 1200), (255, 255, 255)), tmp_path, "w.png")
    assert {i.code for i in check(p, 4, 4).issues} == {"no_alpha"}
    p = save(Image.new("RGBA", (1200, 1200), (255, 255, 255, 255)), tmp_path, "w2.png")
    assert {i.code for i in check(p, 4, 4).issues} == {"no_alpha"}


def test_fully_transparent_file_has_no_new_issue(tmp_path):
    res = check(save(Image.new("RGBA", (1200, 1200), (0, 0, 0, 0)), tmp_path), 4, 4)
    new = {"thin_line", "small_text", "white_box", "non_srgb"}
    assert not new & {i.code for i in res.issues}


# --- non_srgb: criterion 8 --------------------------------------------------------------------


def icc_profile(space: bytes, primaries: dict[bytes, tuple[float, float, float]]) -> bytes:
    """A minimal RGB matrix/TRC ICC v2 profile, enough for the colorants to be read."""

    def s15(v: float) -> bytes:
        return struct.pack(">i", round(v * 65536))

    def xyz(v):
        return b"XYZ \0\0\0\0" + b"".join(s15(c) for c in v)

    desc = b"desc\0\0\0\0" + struct.pack(">I", 5) + b"test\0" + b"\0" * 79
    curv = b"curv\0\0\0\0" + struct.pack(">IH", 1, 0x0233) + b"\0\0"
    tags = {b"desc": desc, b"wtpt": xyz((0.9642, 1.0, 0.8249)), b"cprt": b"text\0\0\0\0t\0\0\0"}
    tags.update({k: xyz(v) for k, v in primaries.items()})
    tags.update({b"rTRC": curv, b"gTRC": curv, b"bTRC": curv})
    table, body, off = b"", b"", 128 + 4 + 12 * len(tags)
    for sig, data in tags.items():
        pad = data + b"\0" * (-len(data) % 4)
        table += sig + struct.pack(">II", off + len(body), len(data))
        body += pad
    size = off + len(body)
    header = struct.pack(">I4sI4s4s4s", size, b"test", 0x02400000, b"mntr", space, b"XYZ ")
    header += b"\0" * 12 + b"acsp" + b"\0" * 24 + struct.pack(">I", 0)
    header += s15(0.9642) + s15(1.0) + s15(0.8249) + b"\0" * 48
    return header + struct.pack(">I", len(tags)) + table + body


ADOBE = {
    b"rXYZ": (0.6097, 0.3111, 0.0195),
    b"gXYZ": (0.2053, 0.6257, 0.0609),
    b"bXYZ": (0.1492, 0.0632, 0.7446),
}
P3 = {
    b"rXYZ": (0.5151, 0.2412, -0.0011),
    b"gXYZ": (0.2920, 0.6922, 0.0419),
    b"bXYZ": (0.1572, 0.0666, 0.7841),
}


def with_profile(tmp_path, profile: bytes | None, name="p.png"):
    im = line_art(20)
    kw = {"icc_profile": profile} if profile else {}
    return save(im, tmp_path, name, **kw)


@pytest.mark.parametrize("colorants", [ADOBE, P3], ids=["adobe_rgb", "display_p3"])
def test_wide_gamut_profile_warns(tmp_path, colorants):
    p = with_profile(tmp_path, icc_profile(b"RGB ", colorants))
    i = issue(check(p, 4, 4), "non_srgb")
    assert i.severity == "warn" and i.regions is None


def test_srgb_and_no_profile_are_fine(tmp_path):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    assert issue(check(with_profile(tmp_path, srgb, "s.png"), 4, 4), "non_srgb") is None
    assert issue(check(with_profile(tmp_path, None, "n.png"), 4, 4), "non_srgb") is None


def test_cmyk_mode_warns(tmp_path):
    p = tmp_path / "c.tif"
    Image.new("CMYK", (800, 800), (10, 200, 200, 0)).save(p)
    assert issue(check(p, 4, 4), "non_srgb")


def test_cmyk_profile_space_is_detected():
    img = pyvips.Image.black(8, 8, bands=3).copy(interpretation="srgb")
    img.set_type(pyvips.GValue.blob_type, "icc-profile-data", icc_profile(b"CMYK", ADOBE))
    assert qa_print.is_non_srgb(img)
    img.set_type(pyvips.GValue.blob_type, "icc-profile-data", icc_profile(b"GRAY", ADOBE))
    assert not qa_print.is_non_srgb(img)


# --- combined, serialization, limits ----------------------------------------------------------


def test_several_issues_are_all_returned(tmp_path):
    p = with_profile(tmp_path, icc_profile(b"RGB ", ADOBE))
    im = line_art(4)
    p = save(im, tmp_path, "m.png", icc_profile=icc_profile(b"RGB ", ADOBE))
    codes = {(i.code, i.severity) for i in check(p, 4, 4).issues}
    assert {("thin_line", "error"), ("non_srgb", "warn")} <= codes


def test_to_dict_has_no_nulls_and_is_json(tmp_path):
    res = check(save(framed((255, 255, 255)), tmp_path), 6, 6)
    d = res.to_dict()
    json.dumps(d)
    box = next(i for i in d["issues"] if i["code"] == "white_box")
    assert "measured_in" not in box and box["regions"]
    plain = next(i for i in check(save(line_art(5), tmp_path, "t.png"), 4, 4).to_dict()["issues"])
    assert set(plain) <= {"code", "severity", "message", "measured_in", "regions"}
    assert None not in plain.values()


def test_file_over_the_decode_cap_is_unreadable_not_a_crash(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "imaging_max_decode_pixels", 10_000)
    res = check(save(line_art(5), tmp_path), 4, 4)
    assert [i.code for i in res.issues] == ["unreadable"]


def test_checks_make_no_network_or_ai_call(monkeypatch, tmp_path):
    import socket

    def boom(*a, **k):
        raise AssertionError("network call from a QA check")

    monkeypatch.setattr(socket.socket, "connect", boom)
    check(save(line_art(5), tmp_path), 4, 4)


# --- criterion 11: a 22 x 60 in design --------------------------------------------------------

LARGE_SCRIPT = """
import json, resource, sys, time
from pathlib import Path
from app.qa import check
t = time.perf_counter()
res = check(Path(sys.argv[1]), 22, 60)
elapsed = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
print(json.dumps({"s": elapsed, "rss_mb": rss_mb, "w": res.width_px, "h": res.height_px,
                  "issues": [i.to_dict() for i in res.issues]}))
"""


def make_large(path):
    """6600 x 18000 px (22 x 60 in at 300 DPI): a blob, a row of letters and a 0.0133 in hairline."""
    w, h = 6600, 18000
    alpha = pyvips.Image.black(w, h)
    alpha = alpha.draw_rect(255, 500, 500, 3000, 3000, fill=True)
    alpha = alpha.draw_rect(255, 500, 9000, 6000, 4, fill=True)  # 4 px hairline = 0.0133 in
    for k in range(6):
        alpha = alpha.draw_rect(255, 800 + k * 40, 12000, 24, 24, fill=True)  # 0.08 in glyphs
    rgb = alpha.new_from_image([20, 100, 200]).copy(interpretation="srgb")
    rgb.bandjoin(alpha).copy(interpretation="srgb").pngsave(str(path), compression=1)
    return path


def test_22_by_60_in_design_is_fast_and_small(tmp_path):
    p = make_large(tmp_path / "big.png")
    out = subprocess.run(
        [sys.executable, "-c", LARGE_SCRIPT, str(p)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert (r["w"], r["h"]) == (6600, 18000)
    by = {i["code"]: i for i in r["issues"]}
    assert by["thin_line"]["severity"] == "error"
    assert by["thin_line"]["measured_in"] == pytest.approx(4 / DPI, abs=0.01)  # 0.009 in pixels
    assert "small_text" in by
    print(f"22x60 in QA: {r['s']:.1f} s, peak RSS {r['rss_mb']:.0f} MB")
    assert r["rss_mb"] < 1500
    assert r["s"] < 20


def test_qa_check_endpoint_carries_the_new_fields(tmp_path):
    import os
    from pathlib import Path

    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app

    dest = Path(os.environ["LOCAL_STORAGE_DIR"]) / "qaprint" / "thin.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    line_art(5).save(dest)
    client = TestClient(app, headers={"X-Imaging-Secret": settings.shared_secrets()[0]})
    body = client.post("/qa/check", json={"file_key": "qaprint/thin.png", "target_width_in": 4})
    assert body.status_code == 200, body.text
    thin = next(i for i in body.json()["issues"] if i["code"] == "thin_line")
    assert thin["severity"] == "error" and 0 < thin["measured_in"] < 0.02
    assert thin["regions"] and set(thin["regions"][0]) == {"left", "top", "width", "height"}
