"""Listing photos (T-26-2): templates, real-size composite, underbase, presets and checks, XMP,
palette and zip."""

import io
import zipfile
from pathlib import Path

import pytest
import pyvips
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import garments, main, photos
from app.config import settings
from app.guard import HEAVY_PATHS
from app.main import app

ROOT = Path(settings.local_storage_dir)
CO = "11111111-2222-4333-8444-555555555555"
OTHER = "99999999-2222-4333-8444-555555555555"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-Imaging-Secret": settings.shared_secrets()[0]}) as c:
        yield c


def put_design(key: str, w: int = 600, h: int = 600, alpha: int = 255, rgb=(230, 60, 40)) -> str:
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((0, 0, w - 1, h - 1), fill=(*rgb, alpha))
    im.save(path)
    return key


@pytest.fixture(scope="module")
def design():
    return put_design(f"{CO}/artwork/2026/10/design.png")


def render(client, **over):
    body = {
        "design_key": f"{CO}/artwork/2026/10/design.png",
        "design_width_in": 10,
        "design_height_in": 8,
        "placement": "front",
        "garment": "tee",
        "view": "front_flat",
        "blank_hex": "#1B3A6B",
        "underbase_preview": False,
        "preset": "amazon_alt",
        "out_key": f"{CO}/photos/out.png",
        "xmp_subjects": [],
        **over,
    }
    return client.post("/photo/render", json=body)


# --- templates ----------------------------------------------------------------------------


def test_sixteen_drawn_templates_with_realistic_print_areas(client):
    rows = client.get("/photo/templates").json()
    assert len(rows) == 16
    assert {(r["garment"], r["view"]) for r in rows} == {
        (g, v) for g in garments.GARMENTS for v in garments.VIEWS
    }
    assert all(r["drawn"] is True for r in rows)
    by = {(r["garment"], r["view"]): r for r in rows}
    assert by[("tee", "front_flat")]["print_areas"] == {"front": {"w_in": 12, "h_in": 16}}
    assert by[("tee", "back")]["print_areas"].keys() == {"back"}
    assert by[("tank", "front_flat")]["print_areas"]["front"]["w_in"] < 12  # narrower
    # Hoodie front print sits above the pocket.
    hood = garments.get_template("hoodie", "front_flat").print_area
    assert hood.y + hood.h <= 15.5


@pytest.mark.parametrize("garment", garments.GARMENTS)
@pytest.mark.parametrize("view", garments.VIEWS)
def test_every_template_has_a_flat_blank_pixel_and_exact_color(garment, view, tmp_path):
    maps = garments.template_maps(garment, view, 600, 600)
    assert maps.flat_px is not None
    _, _, w, _ = maps.print_area_px
    assert w == round(maps.template.print_area.w * maps.ppi)
    art = Path(put_design(f"{CO}/t/{garment}-{view}.png", 60, 60))
    img, _ = photos.compose(maps, ROOT / art, 2, 2, "#1B3A6B", False)
    assert img(*maps.flat_px) == [0x1B, 0x3A, 0x6B]
    assert img(2, 2) == [255, 255, 255]


def test_folds_bend_and_darken_the_print():
    maps = garments.template_maps("tee", "front_flat", 1000, 1000)
    x, y, w, h = maps.print_area_px
    sx, sy = photos._shift(maps.height_map.crop(x, y, w, h), maps.ppi)
    assert max(abs(sx.min()), sx.max(), abs(sy.min()), sy.max()) > 1.0
    assert maps.shadow.crop(x, y, w, h).min() < 240


def test_templates_are_cached_per_size():
    garments.template_maps("crewneck", "back", 640, 640)
    hits = garments.template_maps.cache_info().hits
    garments.template_maps("crewneck", "back", 640, 640)
    assert garments.template_maps.cache_info().hits == hits + 1


# --- real size ------------------------------------------------------------------------------


def test_design_is_placed_at_exact_inches_never_scaled_up(client, design):
    for w_in, h_in in ((10, 8), (2, 2.5)):
        r = render(client, design_width_in=w_in, design_height_in=h_in)
        assert r.status_code == 200, r.text
        body = r.json()
        maps = garments.template_maps("tee", "front_flat", 2000, 2000)
        x, y, w, h = body["print_box_px"]
        assert (w, h) == (round(w_in * maps.ppi), round(h_in * maps.ppi))
        ax, ay, aw, _ = maps.print_area_px
        assert y == ay and abs((x + w / 2) - (ax + aw / 2)) <= 1  # top-centre of print area
        assert (
            body["scale"] == 1 and "design_larger_than_print_area" not in body["checks"]["failures"]
        )
        out = pyvips.Image.new_from_file(str(ROOT / body["key"]))
        # Ink inside the box; the opaque red design covers its center.
        r_, g_ = out(x + w // 2, y + h // 2)[:2]
        assert r_ > 150 and g_ < 90


def test_design_larger_than_print_area_is_scaled_down_and_reported(client, design):
    body = render(client, design_width_in=14, design_height_in=18).json()
    assert body["scale"] == pytest.approx(min(12 / 14, 16 / 18), abs=1e-3)
    assert "design_larger_than_print_area" in body["checks"]["failures"]
    assert body["checks"]["passes"] is False
    _, _, aw, ah = body["print_area_px"]
    assert body["print_box_px"][2] <= aw and body["print_box_px"][3] <= ah


def test_placement_must_match_view(client, design):
    r = render(client, view="back", placement="front")
    assert r.status_code == 422 and "placement back" in r.json()["detail"]
    assert render(client, view="back", placement="back").status_code == 200


# --- underbase ------------------------------------------------------------------------------


def _center(img, maps, design_in=4):
    x, y, dw, dh, _ = photos._place(maps, design_in, design_in)
    return img(x + dw // 2, y + dh // 2)


def test_underbase_preview_on_dark_blank_shows_white_under_soft_pixels():
    maps = garments.template_maps("tee", "front_flat", 800, 800)
    soft = ROOT / put_design(f"{CO}/t/soft.png", 200, 200, alpha=128, rgb=(220, 30, 30))
    plain, _ = photos.compose(maps, soft, 4, 4, "#111111", False)
    under, info = photos.compose(maps, soft, 4, 4, "#111111", True)
    assert info["underbase"] is True
    p, u = _center(plain, maps), _center(under, maps)
    assert u[1] > p[1] + 60 and u[0] > p[0]  # half-transparent red over white is pink


def test_underbase_preview_changes_nothing_on_light_blank():
    maps = garments.template_maps("tee", "front_flat", 800, 800)
    soft = ROOT / put_design(f"{CO}/t/soft2.png", 200, 200, alpha=128, rgb=(220, 30, 30))
    a, _ = photos.compose(maps, soft, 4, 4, "#F2F2F2", False)
    b, info = photos.compose(maps, soft, 4, 4, "#F2F2F2", True)
    assert info["underbase"] is False
    assert (a - b).abs().max() == 0


# --- presets and checks ---------------------------------------------------------------------

PRESET_SIZES = {
    "amazon_main": (2000, 2000),
    "amazon_alt": (2000, 2000),
    "etsy": (2700, 2025),
    "shopify": (2048, 2048),
    "tiktok": (1600, 1600),
    "walmart": (2000, 2000),
}


@pytest.mark.parametrize("preset", list(PRESET_SIZES))
def test_preset_size_format_background_and_checks(client, design, preset):
    body = render(
        client,
        preset=preset,
        out_key=f"{CO}/photos/{preset}.jpg",
        design_width_in=4,
        design_height_in=4,
    ).json()
    assert (body["width_px"], body["height_px"]) == PRESET_SIZES[preset]
    assert body["format"] == "jpeg"
    data = (ROOT / body["key"]).read_bytes()
    assert data[:3] == b"\xff\xd8\xff"
    img = pyvips.Image.new_from_buffer(data, "")
    assert (img.width, img.height, img.bands, img.interpretation) == (
        *PRESET_SIZES[preset],
        3,
        "srgb",
    )
    assert img.get_typeof("icc-profile-data") != 0  # sRGB profile embedded
    c = body["checks"]
    assert c["longest_side_px"] == max(PRESET_SIZES[preset])
    assert c["background_pure_white"] is True
    assert c["fill_ratio"] >= 0.85
    if preset == "amazon_main":
        assert c["failures"] == ["illustration_not_photo"] and c["passes"] is False
    else:
        assert c["failures"] == [] and c["passes"] is True


def test_checks_fail_with_codes():
    maps = garments.template_maps("tank", "front_flat", 400, 400)
    img = (pyvips.Image.black(400, 400, bands=3) + 255).cast("uchar")
    assert photos.background_is_white(img, maps) is True
    grey = img.draw_rect([250, 255, 255], 0, 0, 2, 2, fill=True)
    assert photos.background_is_white(grey, maps) is False
    data = grey.pngsave_buffer()
    strict = photos.Preset(4000, 4000, 4000, white_required=True, min_fill=0.95)
    checks = photos.run_checks(data, maps, strict, too_big=False)
    assert checks.passes is False
    assert checks.failures == ["background_not_white", "fill_below_min", "too_small"]


# --- XMP ----------------------------------------------------------------------------------


@pytest.mark.parametrize("ext", ["jpg", "png"])
def test_xmp_subjects_written_and_absent_when_empty(client, design, ext):
    key = f"{CO}/photos/xmp.{ext}"
    body = render(client, out_key=key, xmp_subjects=["contains-synthetic-performer"]).json()
    data = (ROOT / body["key"]).read_bytes()
    img = pyvips.Image.new_from_buffer(data, "")
    assert img.get_typeof("xmp-data") != 0
    xmp = img.get("xmp-data")
    assert b"<dc:subject>" in xmp and b"<rdf:li>contains-synthetic-performer</rdf:li>" in xmp
    if ext == "png":
        assert Image.open(io.BytesIO(data)).info.get("xmp")

    render(client, out_key=key, xmp_subjects=[])
    data = (ROOT / key).read_bytes()
    assert b"xmpmeta" not in data
    assert pyvips.Image.new_from_buffer(data, "").get_typeof("xmp-data") == 0


def test_xmp_subjects_are_escaped():
    assert b"a&amp;b&lt;c" in photos.xmp_packet(["a&b<c"])


# --- palette --------------------------------------------------------------------------------


def test_palette_shares_and_determinism(client):
    key = f"{CO}/artwork/2026/10/pal.png"
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 99, 49), fill=(250, 250, 250, 255))  # light, 1/4
    d.rectangle((100, 0, 199, 49), fill=(10, 10, 10, 255))  # dark, 1/4
    im.save(path)
    a = client.post("/photo/palette", json={"design_key": key}).json()
    b = client.post("/photo/palette", json={"design_key": key}).json()
    assert a == b
    assert 1 <= len(a["colors"]) <= 6
    assert sum(c["share"] for c in a["colors"]) == pytest.approx(1, abs=0.01)
    assert a["transparent_share"] == pytest.approx(0.5, abs=0.02)
    assert a["light_share"] == pytest.approx(0.5, abs=0.02)
    assert a["dark_share"] == pytest.approx(0.5, abs=0.02)
    assert {c["hex"] for c in a["colors"]} == {"#FAFAFA", "#0A0A0A"}


def test_palette_caps_six_colors():
    path = ROOT / put_design(f"{CO}/t/rainbow.png", 10, 10)
    im = Image.new("RGBA", (120, 10))
    for i in range(12):
        ImageDraw.Draw(im).rectangle(
            (i * 10, 0, i * 10 + 9, 9), fill=(i * 21, 255 - i * 21, (i * 97) % 256, 255)
        )
    im.save(path)
    out = photos.palette(path)
    assert len(out["colors"]) == 6
    assert sum(c["share"] for c in out["colors"]) == pytest.approx(1, abs=0.01)


# --- zip ----------------------------------------------------------------------------------


def test_zip_names_are_sanitised_and_deduplicated(client, design):
    k1 = render(client, out_key=f"{CO}/photos/z1.jpg").json()["key"]
    k2 = render(client, out_key=f"{CO}/photos/z2.jpg").json()["key"]
    items = [
        {"key": k1, "name": "amazon/01-tee.jpg"},
        {"key": k2, "name": "../../etc/passwd"},
        {"key": k2, "name": "/abs/01-tee.jpg"},
        {"key": k1, "name": "amazon/01-tee.jpg"},
    ]
    r = client.post("/photo/zip", json={"items": items, "out_key": f"{CO}/exports/set.zip"})
    assert r.status_code == 200, r.text
    body = r.json()
    data = (ROOT / body["key"]).read_bytes()
    assert body["bytes"] == len(data)
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert names == ["amazon/01-tee.jpg", "etc/passwd", "abs/01-tee.jpg", "amazon/01-tee-2.jpg"]
    assert all(".." not in n and not n.startswith("/") for n in names)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.read("amazon/01-tee.jpg") == (ROOT / k1).read_bytes()


def put_raw(key: str, n_bytes: int) -> str:
    """A file of exactly `n_bytes`, for exercising the zip byte caps without a real 50 MB fixture
    (S-50): the caps are checked from storage size before any content is read."""
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(n_bytes)
    return key


def test_zip_rejects_an_item_over_the_per_item_cap(client, monkeypatch):
    monkeypatch.setattr(main, "PHOTO_ZIP_MAX_ITEM_BYTES", 1000)
    big = put_raw(f"{CO}/photos/big.jpg", 1001)
    r = client.post(
        "/photo/zip",
        json={"items": [{"key": big, "name": "big.jpg"}], "out_key": f"{CO}/exports/cap.zip"},
    )
    assert r.status_code == 422 and "per-item limit" in r.json()["detail"]


def test_zip_rejects_items_over_the_total_cap(client, monkeypatch):
    monkeypatch.setattr(main, "PHOTO_ZIP_MAX_TOTAL_BYTES", 1500)
    a = put_raw(f"{CO}/photos/a.jpg", 1000)
    b = put_raw(f"{CO}/photos/b.jpg", 1000)
    r = client.post(
        "/photo/zip",
        json={
            "items": [{"key": a, "name": "a.jpg"}, {"key": b, "name": "b.jpg"}],
            "out_key": f"{CO}/exports/total.zip",
        },
    )
    assert r.status_code == 422 and "total limit" in r.json()["detail"]


def test_zip_rejects_a_disallowed_extension(client):
    csv_key = put_raw(f"{CO}/uploads/orders.csv", 10)
    r = client.post(
        "/photo/zip",
        json={"items": [{"key": csv_key, "name": "orders.csv"}], "out_key": f"{CO}/exports/x.zip"},
    )
    assert r.status_code == 422 and ".jpg, .jpeg or .png" in r.json()["detail"]


# --- safety ---------------------------------------------------------------------------------


def test_keys_outside_the_company_prefix_are_400(client, design):
    put_design(f"{OTHER}/artwork/x.png")
    r = render(client, design_key=f"{OTHER}/artwork/x.png")
    assert r.status_code == 400 and "company prefix" in r.json()["detail"]
    r = render(client, out_key="photos/no-company.png")
    assert r.status_code == 400
    r = client.post(
        "/photo/zip",
        json={
            "items": [{"key": f"{OTHER}/artwork/x.png", "name": "x.png"}],
            "out_key": f"{CO}/exports/x.zip",
        },
    )
    assert r.status_code == 400 and "items[0].key" in r.json()["detail"]
    r = client.post("/photo/palette", json={"design_key": "../etc/passwd"})
    assert r.status_code == 400


def test_photo_routes_need_the_secret_and_share_the_heavy_limit():
    with TestClient(app) as anon:
        assert anon.get("/photo/templates").status_code == 401
        assert anon.post("/photo/render", json={}).status_code == 401
        assert anon.post("/photo/zip", json={}).status_code == 401
        assert anon.post("/photo/palette", json={}).status_code == 401
    assert {"/photo/render", "/photo/zip"} <= HEAVY_PATHS


def test_bad_output_extension_and_unreadable_design_are_422(client, design):
    r = render(client, out_key=f"{CO}/photos/out.gif")
    assert r.status_code == 422 and ".jpg" in r.json()["detail"]
    bad = ROOT / CO / "artwork" / "fake.png"
    bad.write_bytes(b"not a png at all")
    r = render(client, design_key=f"{CO}/artwork/fake.png")
    assert r.status_code == 422
