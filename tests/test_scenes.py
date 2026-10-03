"""AI lifestyle scenes (T-27-2): scene base and edit mask, scene composite with registration,
check 1 (garment outline aligned), real-size print with the scene's light, check 2 (design lock),
presets, XMP, and the safety of untrusted model output."""

from pathlib import Path

import pytest
import pyvips
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from scene_helpers import base_image, drawn_marks, folds, moved_garment, perturbed

from app import main, photos, scenes
from app.config import settings
from app.fonts import parse_hex
from app.garments import template_maps
from app.guard import HEAVY_PATHS
from app.main import app

ROOT = Path(settings.local_storage_dir)
CO = "11111111-2222-4333-8444-555555555555"
OTHER = "99999999-2222-4333-8444-555555555555"
NAVY, WHITE = "#1B3A6B", "#F2F2F2"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-Imaging-Secret": settings.shared_secrets()[0]}) as c:
        yield c


def put_logo(key: str, rgb=(230, 60, 40), kind: str = "logo") -> str:
    """A design with structure: a disc with a block and a small mark, or stripes."""
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if kind == "logo":
        d.ellipse((40, 40, 560, 560), fill=(*rgb, 255))
        d.rectangle((200, 150, 400, 450), fill=(250, 240, 60, 255))
        d.rectangle((260, 470, 340, 500), fill=(20, 20, 120, 255))
    else:
        for i in range(6):
            d.rectangle((i * 100, 0, i * 100 + 50, 600), fill=(40, 90, 230, 255))
    im.save(path)
    return key


DESIGN = f"{CO}/artwork/2026/10/scene-logo.png"


@pytest.fixture(scope="module", autouse=True)
def design():
    put_logo(DESIGN)
    put_logo(f"{CO}/artwork/2026/10/scene-stripes.png", kind="stripes")
    put_logo(f"{CO}/artwork/2026/10/scene-green.png", rgb=(40, 170, 60))


def make_base(client, garment, view, blank, size=1024, name="base"):
    r = client.post(
        "/photo/scene-base",
        json={
            "garment": garment,
            "view": view,
            "blank_hex": blank,
            "size_px": size,
            "out_key": f"{CO}/photos/scenes/{name}.png",
            "mask_out_key": f"{CO}/photos/scenes/{name}-mask.png",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def put_scene(img: pyvips.Image, name: str, fmt: str = "png") -> str:
    key = f"{CO}/photos/scenes/raw/{name}.{fmt}"
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        img.pngsave(str(path))
    else:
        img.jpegsave(str(path), Q=90)
    return key


def composite(client, base, scene_key, **over):
    body = {
        "scene_key": scene_key,
        "base_key": base["key"],
        "print_box_px": base["print_box_px"],
        "design_key": DESIGN,
        "design_width_in": 8,
        "design_height_in": 8,
        "placement": "front",
        "blank_hex": NAVY,
        "preset": "shopify",
        "out_key": f"{CO}/photos/scenes/out/{scene_key.rsplit('/', 1)[-1].split('.')[0]}.jpg",
        "xmp_subjects": ["contains-synthetic-performer"],
        **over,
    }
    return client.post("/photo/scene-composite", json=body)


# --- scene base -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "garment,view,blank,size",
    [
        ("tee", "on_model_white", NAVY, 1024),
        ("hoodie", "lifestyle_base", WHITE, [1024, 1536]),
        ("tee", "lifestyle_base", WHITE, [1536, 1024]),
    ],
)
def test_scene_base_and_mask_in_the_provider_convention(client, garment, view, blank, size):
    body = make_base(client, garment, view, blank, size, name=f"b-{garment}-{view}")
    w, h = (size, size) if isinstance(size, int) else size
    base = pyvips.Image.new_from_file(str(ROOT / body["key"]))
    mask = pyvips.Image.new_from_file(str(ROOT / body["mask_key"]))
    assert (base.width, base.height) == (w, h) == (body["width_px"], body["height_px"])
    assert (mask.width, mask.height, mask.bands) == (w, h, 4)  # PNG with alpha, same size

    maps = template_maps(garment, view, w, h)
    assert body["print_box_px"] == list(maps.print_area_px)
    # The print area (real inches at the template's ppi) plus the margin is opaque = protected.
    x, y, bw, bh = body["print_box_px"]
    assert bw == round(maps.template.print_area.w * maps.ppi)
    px, py, pw, ph = body["protected_box_px"]
    m = round(scenes.MASK_MARGIN_IN * maps.ppi)
    assert (px, py, pw, ph) == (x - m, y - m, bw + 2 * m, bh + 2 * m)
    alpha = mask[3]
    assert alpha.crop(px, py, pw, ph).min() == 255
    assert alpha.crop(0, 0, w, py).max() == 0 and alpha.crop(0, py + ph, w, h - py - ph).max() == 0
    # The blank is exactly blank_hex on an unshaded garment pixel; no design anywhere.
    assert base(*maps.flat_px)[:3] == list(parse_hex(blank)[:3])
    corner = base(2, 2)[:3]
    assert corner == ([255] * 3 if view == "on_model_white" else list(scenes.LIFESTYLE_BG))
    assert scenes.read_base_meta(ROOT / body["key"]) == (garment, view)


def test_lifestyle_base_leaves_room_for_a_scene():
    on_model = template_maps("tee", "on_model_white", 1024, 1024)
    life = template_maps("tee", "lifestyle_base", 1024, 1024)
    assert life.product_bbox_px[2] < 0.75 * on_model.product_bbox_px[2]
    assert life.figure is not None  # a neutral figure, as on_model_white


def test_scene_base_validation(client):
    ok = {
        "garment": "tee",
        "view": "on_model_white",
        "blank_hex": NAVY,
        "out_key": f"{CO}/photos/scenes/v.png",
        "mask_out_key": f"{CO}/photos/scenes/v-mask.png",
    }
    r = client.post("/photo/scene-base", json={**ok, "view": "folded"})
    assert r.status_code == 422
    r = client.post("/photo/scene-base", json={**ok, "size_px": 4096})
    assert r.status_code == 422 and "size_px" in r.json()["detail"]
    r = client.post("/photo/scene-base", json={**ok, "out_key": f"{CO}/photos/scenes/v.jpg"})
    assert r.status_code == 422 and ".png" in r.json()["detail"]
    r = client.post("/photo/scene-base", json={**ok, "mask_out_key": f"{OTHER}/photos/m.png"})
    assert r.status_code == 400 and "company prefix" in r.json()["detail"]


# --- composite: passes ----------------------------------------------------------------------


def test_unchanged_scene_passes_both_checks_and_prints_real_size(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="same")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "same")
    r = composite(client, base, scene)
    assert r.status_code == 200, r.text
    body = r.json()
    c = body["checks"]
    assert c["passes"] is True and c["failures"] == []
    assert c["region_unchanged_score"] >= 0.95 and c["design_lock_score"] >= 0.95
    assert c["region_unchanged_min"] == scenes.REGION_UNCHANGED_MIN
    assert c["design_lock_min"] == scenes.DESIGN_LOCK_MIN
    assert body["key"] is not None and (body["width_px"], body["height_px"]) == (2048, 2048)
    assert body["upscale"] == pytest.approx(2.0, abs=0.01) and body["upscale_method"] == "lbb"

    data = (ROOT / body["key"]).read_bytes()
    out = pyvips.Image.new_from_buffer(data, "")
    assert (out.width, out.height, out.bands) == (2048, 2048, 3)
    assert b"<rdf:li>contains-synthetic-performer</rdf:li>" in out.get("xmp-data")
    # The design covers exactly 8 x 8 in at the output's ppi; its center is the logo's block.
    maps = template_maps("tee", "on_model_white", 2048, 2048)
    x, y, w, h = body["print_box_px"]
    assert (w, h) == (round(8 * maps.ppi), round(8 * maps.ppi))
    r_, g_, b_ = out(x + w // 2, y + h // 2)[:3]
    assert r_ > 200 and g_ > 190 and b_ < 120  # yellow block
    # Deterministic: the same call writes the same bytes (a retried job overwrites).
    assert composite(client, base, scene).status_code == 200
    assert (ROOT / body["key"]).read_bytes() == data


@pytest.mark.parametrize(
    "garment,view,blank,size,provider,preset",
    [
        ("tee", "on_model_white", NAVY, 1024, (1024, 1024), "etsy"),
        ("hoodie", "lifestyle_base", WHITE, [1024, 1536], (1024, 1536), "shopify"),
        ("tee", "lifestyle_base", WHITE, 1024, (1536, 1024), "amazon_alt"),
    ],
)
def test_perturbed_whole_frame_still_passes(client, garment, view, blank, size, provider, preset):
    """New background, relit and tinted, noise, 2% scale and a shift, a provider size, JPEG."""
    base = make_base(client, garment, view, blank, size, name=f"p-{garment}-{view}")
    img = base_image(garment, view, base["width_px"], base["height_px"], blank)
    scene = put_scene(perturbed(img, garment, view, provider), f"p-{garment}-{view}", "jpg")
    body = composite(client, base, scene, blank_hex=blank, preset=preset).json()
    c = body["checks"]
    assert c["passes"] is True, body
    assert c["region_unchanged_score"] >= 0.85 and c["design_lock_score"] >= 0.9
    reg = body["registration"]
    assert reg["scale"] == pytest.approx(1.02, abs=0.006)
    want = photos.PRESETS[preset]
    assert (body["width_px"], body["height_px"]) == (want.width, want.height)
    assert max(want.width, want.height) >= 2000  # every scene preset is >= 2000 px
    out = pyvips.Image.new_from_file(str(ROOT / body["key"]))
    assert (out.width, out.height) == (want.width, want.height)


def test_folds_and_light_in_the_scene_still_pass_check_2(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="folds")
    img = base_image("tee", "on_model_white", 1024, 1024, NAVY)
    shaded = folds(perturbed(img, "tee", "on_model_white", (1024, 1024)), base["print_box_px"], 0.3)
    body = composite(client, base, put_scene(shaded, "folds")).json()
    assert body["checks"]["passes"] is True, body
    assert body["checks"]["design_lock_score"] < 0.97  # the folds really are on the print


# --- composite: check 1 fails ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name,make",
    [
        ("moved", lambda: moved_garment("tee", "lifestyle_base", 1024, 1024, WHITE, dx_in=0.6)),
        (
            "moved-y",
            lambda: moved_garment("hoodie", "lifestyle_base", 1024, 1024, WHITE, dy_in=0.6),
        ),
        ("widened", lambda: moved_garment("tee", "on_model_white", 1024, 1024, WHITE, widen=1.1)),
        (
            "far",
            lambda: perturbed(
                base_image("tee", "lifestyle_base", 1024, 1024, WHITE),
                "tee",
                "lifestyle_base",
                (1024, 1024),
                shift=(150, 0),
            ),
        ),
    ],
)
def test_moved_or_reshaped_garment_fails_check_1_and_saves_nothing(client, name, make):
    view = "on_model_white" if name == "widened" else "lifestyle_base"
    garment = "hoodie" if name == "moved-y" else "tee"
    base = make_base(client, garment, view, WHITE, name=f"f-{name}")
    scene = put_scene(make(), f"f-{name}")
    out_key = f"{CO}/photos/scenes/out/fail-{name}.jpg"
    body = composite(client, base, scene, blank_hex=WHITE, out_key=out_key).json()
    c = body["checks"]
    assert c["passes"] is False and c["failures"] == ["region_changed"], body
    assert (
        c["region_unchanged_score"] < scenes.REGION_UNCHANGED_MIN
        or body["registration"]["at_limit"]
    )
    assert c["design_lock_score"] is None
    assert body["key"] is None and not (ROOT / out_key).exists()


# --- composite: check 1b (something drawn into the print box) -------------------------------


@pytest.mark.parametrize(
    "garment,view,blank,kind",
    [
        ("tee", "on_model_white", NAVY, "logo"),
        ("tee", "on_model_white", WHITE, "logo"),
        ("hoodie", "lifestyle_base", NAVY, "logo"),
        ("tee", "on_model_white", WHITE, "wordmark"),
    ],
)
def test_provider_art_in_the_print_box_fails_and_saves_nothing(client, garment, view, blank, kind):
    """The realistic 'mask ignored' case: art inside the print box, the garment outline intact,
    so check 1 alone passes. Nothing is restored over it and nothing is written."""
    base = make_base(client, garment, view, blank, name=f"art-{garment}-{blank[1:]}")
    img = base_image(garment, view, 1024, 1024, blank)
    ink = (40, 40, 40) if blank == WHITE else (230, 230, 230)
    sc = drawn_marks(perturbed(img, garment, view, (1024, 1024)), base["print_box_px"], ink, kind)
    name = f"art-{kind}-{garment}-{blank[1:]}"
    out_key = f"{CO}/photos/scenes/out/{name}.jpg"
    body = composite(client, base, put_scene(sc, name, "jpg"), blank_hex=blank, out_key=out_key)
    body = body.json()
    c = body["checks"]
    assert c["region_unchanged_score"] >= scenes.REGION_UNCHANGED_MIN  # the outline is intact
    assert c["passes"] is False and c["failures"] == ["region_changed"], body
    assert c["design_lock_score"] is None
    assert body["key"] is None and not (ROOT / out_key).exists()


@pytest.mark.parametrize("blank", [NAVY, WHITE, "#111111", "#C0302A"])
def test_print_box_check_ignores_light_folds_and_noise(blank):
    """Relit, tinted, noisy, JPEG'd scenes with strong folds over the print stay at 0; drawn
    marks are well above the limit on the same blank."""
    maps = template_maps("tee", "on_model_white", 1024, 1024)
    img = base_image("tee", "on_model_white", 1024, 1024, blank)
    p = perturbed(img, "tee", "on_model_white", (1024, 1024))

    def foreign(sc):
        sc = pyvips.Image.new_from_buffer(sc.jpegsave_buffer(Q=85), "")
        return scenes.print_box_foreign(sc, img, maps, scenes.register(sc, img, maps))

    for clean in (p, folds(p, maps.print_area_px, 0.15), folds(p, maps.print_area_px, 0.3)):
        assert foreign(clean) <= scenes.BOX_FOREIGN_MAX
    ink = (40, 40, 40) if parse_hex(blank)[0] > 128 else (230, 230, 230)
    mark = drawn_marks(p, maps.print_area_px, ink, "wordmark")
    assert foreign(mark) > 2 * scenes.BOX_FOREIGN_MAX


def test_scene_light_on_the_print_is_low_frequency_and_bounded():
    """Whatever the scene has in the print box, only broad light reaches the restored blank:
    text-like bars at full contrast leave a shade field inside SHADE_RANGE with little of
    their structure."""
    w = h = 400
    ppi = 50.0
    blank = pyvips.Image.black(w, h, bands=3).cast("float") + 128
    bars = pyvips.Image.black(w, h, bands=3) + 128
    for i in range(6):
        bars = bars.draw_rect([80, 80, 80], 40, 40 + i * 50, 320, 16, fill=True)  # 0.32 in bars
    raw = 1 - 80 / 128  # the bars' own contrast against the blank
    shade = scenes._shade_field(bars.cast("float"), blank, scenes.SHADE_SIGMA_IN * ppi).numpy()
    lo, hi = scenes.SHADE_RANGE
    assert shade.min() >= lo - 1e-6 and shade.max() <= hi + 1e-6
    for b in (10, 10, 10), (250, 250, 250):  # and any contrast is clamped
        hard = pyvips.Image.black(w, h, bands=3).cast("float") + b
        s = scenes._shade_field(hard, blank, scenes.SHADE_SIGMA_IN * ppi).numpy()
        assert s.min() >= lo - 1e-6 and s.max() <= hi + 1e-6
    col = shade[90:260, 200]  # across the bars, inside the block: the stripes are gone
    assert col.max() - col.min() < 0.1 * raw


@pytest.mark.parametrize("blank", [NAVY, WHITE])
def test_marks_drawn_over_the_print_lower_check_2(blank):
    """Check 2 compares the printed region with the source design: marks over it fail."""
    out_scene, maps, crop, gain = _scene_frame("tee", "on_model_white", blank)
    rgb = tuple(parse_hex(blank)[:3])
    source = ROOT / DESIGN
    out = scenes._restore_and_print(out_scene, maps, crop, gain, source, 8, 8, rgb, False)
    assert scenes.lock_check(out, maps, crop, source, 8, 8, rgb, False) >= 0.95
    x, y, dw, dh, _ = photos._place(maps, 8, 8)
    for ink in ((40, 40, 40), (230, 230, 230)):
        marked = drawn_marks(out, (x - crop[0], y - crop[1], dw, dh), ink)
        assert scenes.lock_check(marked, maps, crop, source, 8, 8, rgb, False) < 0.75


# --- check 2 fails --------------------------------------------------------------------------


def _scene_frame(garment, view, blank):
    """Everything composite_scene does up to printing, for a perturbed scene."""
    img = base_image(garment, view, 1024, 1024, blank)
    maps_b = template_maps(garment, view, 1024, 1024)
    sc = perturbed(img, garment, view, (1024, 1024))
    reg = scenes.register(sc, img, maps_b)
    c1 = scenes.outline_check(sc, img, maps_b, reg)
    maps, (cx, cy), k, e = scenes._frame(photos.PRESETS["shopify"], maps_b, garment, view)
    m, off = reg.scene_to(k, (e[0] - cx, e[1] - cy))
    return scenes._warp(sc, m, off, 2048, 2048), maps, (cx, cy), c1.gain


@pytest.mark.parametrize("blank", [NAVY, WHITE])
def test_swapped_or_recolored_design_fails_check_2(blank):
    out_scene, maps, crop, gain = _scene_frame("tee", "on_model_white", blank)
    rgb = tuple(parse_hex(blank)[:3])
    source = ROOT / DESIGN
    scores = {}
    for name in ("scene-logo", "scene-stripes", "scene-green"):
        printed = ROOT / f"{CO}/artwork/2026/10/{name}.png"
        out = scenes._restore_and_print(out_scene, maps, crop, gain, printed, 8, 8, rgb, False)
        scores[name] = scenes.lock_check(out, maps, crop, source, 8, 8, rgb, False)
    assert scores["scene-logo"] >= 0.95
    assert scores["scene-stripes"] < 0.5  # swapped
    assert scores["scene-green"] < scenes.DESIGN_LOCK_MIN - 0.05  # recolored disc


def test_design_drift_fails_and_saves_nothing(client, monkeypatch):
    """When the printed region doesn't match the source, nothing is written: here the
    compositor prints another design than the one checked."""
    base = make_base(client, "tee", "on_model_white", NAVY, name="drift")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "drift")
    real = scenes._restore_and_print
    stripes = ROOT / f"{CO}/artwork/2026/10/scene-stripes.png"

    def wrong_design(out_scene, maps, crop, gain, design, *rest):
        return real(out_scene, maps, crop, gain, stripes, *rest)

    monkeypatch.setattr(scenes, "_restore_and_print", wrong_design)
    out_key = f"{CO}/photos/scenes/out/drift.jpg"
    body = composite(client, base, scene, out_key=out_key).json()
    c = body["checks"]
    assert c["passes"] is False and c["failures"] == ["design_drift"]
    assert c["design_lock_score"] < scenes.DESIGN_LOCK_MIN
    assert body["key"] is None and not (ROOT / out_key).exists()


# --- presets and inputs ---------------------------------------------------------------------


def test_lifestyle_is_never_amazon_main_and_white_presets_report_it(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="presets")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "presets")
    r = composite(client, base, scene, preset="amazon_main")
    assert r.status_code == 422 and "amazon_alt" in r.json()["detail"]
    body = composite(client, base, scene, preset="walmart").json()
    assert body["preset_failures"] == ["background_not_white"]


def test_print_box_must_match_the_base(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="box")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "box")
    x, y, w, h = base["print_box_px"]
    r = composite(client, base, scene, print_box_px=[x + 30, y, w, h])
    assert r.status_code == 422 and "print area" in r.json()["detail"]


def _raw(key: str, data: bytes) -> str:
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return key


def test_untrusted_scene_is_refused_with_400(client, monkeypatch):
    base = make_base(client, "tee", "on_model_white", NAVY, name="unsafe")
    raw = f"{CO}/photos/scenes/raw"
    cases = {
        "not an image": _raw(f"{raw}/text.png", b"hello, not an image"),
        "svg": _raw(f"{raw}/vec.svg", b'<svg xmlns="http://www.w3.org/2000/svg" width="9"/>'),
        "renamed": put_scene(pyvips.Image.black(300, 300, bands=3), "renamed", "jpg"),
        "too big": put_scene(pyvips.Image.black(4100, 256, bands=3), "wide"),
        "too small": put_scene(pyvips.Image.black(200, 200, bands=3), "tiny"),
    }
    renamed = ROOT / cases["renamed"]
    cases["renamed"] = _raw(f"{raw}/renamed.png", renamed.read_bytes())  # JPEG bytes as .png
    for what, key in cases.items():
        r = composite(client, base, key)
        assert r.status_code == 400, (what, r.text)
        assert r.json()["detail"].startswith("scene"), (what, r.json())

    monkeypatch.setattr(scenes, "SCENE_MAX_BYTES", 1000)
    big = put_scene(pyvips.Image.black(300, 300, bands=3) + 128, "bytes")
    r = composite(client, base, big)
    assert r.status_code == 400 and "MB" in r.json()["detail"]


def _put_base(w: int, h: int, name: str) -> str:
    """A `base_key`-shaped PNG not made by `render_base` (no scene-base metadata), at an
    arbitrary size: any PNG a company owns passes `check_scope`, same as a real base (S-54)."""
    key = f"{CO}/photos/scenes/raw/{name}.png"
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    pyvips.Image.black(w, h, bands=3).pngsave(str(path))
    return key


def test_oversized_or_undersized_base_is_refused_with_400(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="basesize")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "basesize")

    # A 9000x9000 base (same shape as the probe that drove the server to 2.6 GB RSS) is refused
    # from the header alone, before any decode or template draw.
    huge = {**base, "key": _put_base(9000, 9000, "huge-base")}
    r = composite(client, huge, scene)
    assert r.status_code == 400 and r.json()["detail"].startswith("base is 9000 x 9000")
    assert "2048" in r.json()["detail"]

    tiny = {**base, "key": _put_base(200, 200, "tiny-base")}
    r = composite(client, tiny, scene)
    assert r.status_code == 400 and r.json()["detail"].startswith("base is 200 x 200")
    assert "256" in r.json()["detail"]

    # A normal, render_base-sized base still composites.
    assert composite(client, base, scene).status_code == 200


def test_keys_outside_the_company_are_400(client):
    base = make_base(client, "tee", "on_model_white", NAVY, name="scope")
    scene = put_scene(pyvips.Image.new_from_file(str(ROOT / base["key"]))[0:3], "scope")
    other = _raw(f"{OTHER}/photos/x.png", (ROOT / scene).read_bytes())
    assert composite(client, base, other).status_code == 400
    assert composite(client, base, scene, design_key=f"{OTHER}/artwork/x.png").status_code == 400
    assert composite(client, {**base, "key": f"{OTHER}/b.png"}, scene).status_code == 400


def test_scene_routes_need_the_secret_and_share_the_heavy_limit():
    with TestClient(app) as anon:
        assert anon.post("/photo/scene-base", json={}).status_code == 401
        assert anon.post("/photo/scene-composite", json={}).status_code == 401
    assert {"/photo/scene-base", "/photo/scene-composite"} <= HEAVY_PATHS
    assert main.photo_scene_composite is not None
