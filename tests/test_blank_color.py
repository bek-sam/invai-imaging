"""Blank colour in lifestyle scenes (T-49-5, B-287, architect ruling R4): CIEDE2000 in numpy,
the measurement, the tint fit, the warning (never a failure), and the drawstrings."""

from pathlib import Path

import numpy as np
import pytest
import pyvips
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from scene_helpers import perturbed

from app import photos, scenes
from app.config import settings
from app.garments import template_maps
from app.main import app

# Sharma, Wu, Dalal (2005), "The CIEDE2000 Color-Difference Formula", Table 1: 34 pairs.
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def test_ciede2000_matches_the_sharma_2005_table_to_1e_4():
    assert len(SHARMA) == 34
    got = np.array([float(scenes.ciede2000(a, b)) for a, b, _ in SHARMA])
    want = np.array([e for _, _, e in SHARMA])
    assert np.abs(got - want).max() < 1e-4, got - want


def test_ciede2000_is_vectorised_and_symmetric_enough():
    a = np.array([s[0] for s in SHARMA])
    b = np.array([s[1] for s in SHARMA])
    assert scenes.ciede2000(a, b).shape == (34,)
    assert np.allclose(scenes.ciede2000(a, b), scenes.ciede2000(b, a), atol=1e-6)


# --- the measurement, the fit and the warning ------------------------------------------------


SEED_BLANKS = {"white": "#f7f7f5", "black": "#1a1a1a", "heather": "#b7b9ba", "red": "#a4343a"}
ROOT = Path(settings.local_storage_dir)
CO = "11111111-2222-4333-8444-555555555555"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-Imaging-Secret": settings.shared_secrets()[0]}) as c:
        yield c


@pytest.fixture(scope="module")
def logo(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("blank") / "logo.png"
    im = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 560, 560), fill=(230, 60, 40, 255))
    d.rectangle((200, 150, 400, 450), fill=(250, 240, 60, 255))
    im.save(path)
    return path


@pytest.fixture
def no_check1(monkeypatch):
    """The synthetic backdrop of a grey blank can trip check 1 (a known sensitivity, T-38-1) on
    scenes that are fine for this card's question: what colour the restored blank has."""
    monkeypatch.setattr(scenes, "region_reasons", lambda *a: [])


def run_scene(tmp_path, logo, garment, view, hex_, w_in=8.0, **perturb):
    res = scenes.render_base(garment, view, 1024, 1024, hex_)
    bp = tmp_path / "base.png"
    bp.write_bytes(res.png)
    base = pyvips.Image.new_from_buffer(res.png, "")[0:3]
    sc = perturbed(base, garment, view, (1024, 1024), scale=1.0, shift=(0, 0), **perturb)
    sp = tmp_path / "scene.png"
    sc.pngsave(str(sp))
    maps_b = template_maps(garment, view, base.width, base.height)
    return scenes.composite_scene(
        sp, bp, maps_b.print_area_px, logo, w_in, w_in, hex_, "shopify", "png", []
    )


MOCK_LIKE = {"gain": 0.985, "tint": (1, 1, 1), "noise": 1.5}
WARM = {"gain": 0.92, "tint": (1.0, 0.97, 0.92), "noise": 3.0}
STRONG_WARM = {"gain": 0.9, "tint": (1.0, 0.92, 0.80), "noise": 3.0}


def test_blank_shift_counts_hue_and_chroma_not_lightness():
    lin = scenes.srgb_to_linear
    ref = (247, 247, 245)
    # the same colour at 60% light (a uniform shade in linear light) is not a shift ...
    assert scenes.blank_shift(lin(np.array(ref)) * 0.6, ref) < 0.05
    # ... a warm cast at the same luminance is
    warm = lin(np.array([255, 235, 205]))
    warm = warm * (scenes._SRGB_TO_XYZ[1] @ lin(np.array(ref))) / (scenes._SRGB_TO_XYZ[1] @ warm)
    assert scenes.blank_shift(warm, ref) > scenes.BLANK_DELTA_E_WARN


@pytest.mark.parametrize("name", SEED_BLANKS)
def test_mock_like_scene_keeps_every_seed_blank_true(tmp_path, logo, no_check1, name):
    r = run_scene(tmp_path, logo, "tee", "lifestyle_base", SEED_BLANKS[name], **MOCK_LIKE)
    assert r.passes and r.data is not None and r.failures == []
    assert r.blank_delta_e is not None and r.blank_delta_e < 1.5
    assert r.warnings == []


@pytest.mark.parametrize("name", SEED_BLANKS)
@pytest.mark.parametrize("light", [WARM, STRONG_WARM], ids=["warm", "strong_warm"])
def test_warm_scene_does_not_recolour_the_blank(tmp_path, logo, no_check1, name, light):
    r = run_scene(tmp_path, logo, "tee", "on_model_white", SEED_BLANKS[name], **light)
    assert r.passes and r.failures == [] and r.data is not None
    assert r.blank_delta_e is not None and r.blank_delta_e <= 2.3, r.blank_delta_e
    assert r.warnings == []


def test_without_the_tint_fit_a_warm_white_tee_is_visibly_recoloured(
    tmp_path, logo, no_check1, monkeypatch
):
    monkeypatch.setattr(scenes, "BLANK_TINT_DE_MAX", 1e9)  # the fit accepts the first guess
    r = run_scene(tmp_path, logo, "tee", "on_model_white", SEED_BLANKS["white"], **WARM)
    assert r.blank_delta_e > scenes.BLANK_DELTA_E_WARN
    # A warning only: the render stays usable, `passes` and `failures` don't move (R4).
    assert r.warnings == ["blank_color_shift"]
    assert r.passes is True and r.failures == [] and r.data is not None


def test_a_uniformly_darker_scene_is_not_a_colour_shift(tmp_path, logo, no_check1):
    r = run_scene(
        tmp_path, logo, "tee", "on_model_white", SEED_BLANKS["white"], gain=0.6, tint=(1, 1, 1)
    )
    assert r.blank_delta_e < 1.0 and r.warnings == []


def test_scene_composite_reports_the_new_optional_fields(client, logo):
    key = f"{CO}/artwork/2026/10/blank-logo.png"
    (ROOT / key).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / key).write_bytes(logo.read_bytes())
    r = client.post(
        "/photo/scene-base",
        json={
            "garment": "tee",
            "view": "on_model_white",
            "blank_hex": "#F7F7F5",
            "size_px": 1024,
            "out_key": f"{CO}/photos/scenes/bc-base.png",
            "mask_out_key": f"{CO}/photos/scenes/bc-mask.png",
        },
    )
    assert r.status_code == 200, r.text
    base = r.json()
    img = pyvips.Image.new_from_buffer((ROOT / base["key"]).read_bytes(), "")[0:3]
    scene = perturbed(img, "tee", "on_model_white", (1024, 1024), **WARM)
    skey = f"{CO}/photos/scenes/raw/bc-scene.png"
    (ROOT / skey).parent.mkdir(parents=True, exist_ok=True)
    scene.pngsave(str(ROOT / skey))
    body = client.post(
        "/photo/scene-composite",
        json={
            "scene_key": skey,
            "base_key": base["key"],
            "print_box_px": base["print_box_px"],
            "design_key": key,
            "design_width_in": 8,
            "design_height_in": 8,
            "placement": "front",
            "blank_hex": "#F7F7F5",
            "preset": "shopify",
            "out_key": f"{CO}/photos/scenes/out/bc.jpg",
            "xmp_subjects": [],
        },
    ).json()
    c = body["checks"]
    assert c["passes"] is True, body
    assert isinstance(c["blank_delta_e"], float) and c["blank_delta_e"] <= 2.3
    assert c["warnings"] == []
    assert c["failures"] == []


# --- drawstrings ----------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["lifestyle_base", "on_model_white"])
def test_hoodie_drawstrings_lie_under_an_opaque_print(tmp_path, no_check1, view):
    solid = tmp_path / "solid.png"
    Image.new("RGBA", (600, 600), (200, 40, 40, 255)).save(solid)
    r = run_scene(tmp_path, solid, "hoodie", view, SEED_BLANKS["white"], **MOCK_LIKE)
    assert r.data is not None
    maps_b = template_maps("hoodie", view, 1024, 1024)
    maps, (cx, cy), _, _ = scenes._frame(photos.PRESETS["shopify"], maps_b, "hoodie", view)
    luma = pyvips.Image.new_from_buffer(r.data, "")[0:3].numpy().astype(float) @ [
        0.2126,
        0.7152,
        0.0722,
    ]
    _, by, _, bh = r.print_box_px
    ppi, x_ctr = maps.ppi, maps.to_px(0, 0)[0] - cx
    rows = slice(
        max(by, int(maps.to_px(0, 4.2)[1] - cy)), min(by + bh, int(maps.to_px(0, 10.5)[1] - cy))
    )
    assert rows.stop - rows.start > 40  # the strings really cross this print box

    def col(x_in):
        return round(x_ctr + x_in * ppi)

    for s in (1, -1):
        ref = 0.5 * (luma[rows, col(s * 0.8)].mean() + luma[rows, col(s * 2.2)].mean())
        for c in range(col(s * 1.4) - 4, col(s * 1.4) + 5):
            assert abs(luma[rows, c].mean() - ref) < 2.0  # was ~11 levels with the strings over it


def test_template_maps_keep_the_strings_in_the_blank_but_not_for_the_print():
    m = template_maps("hoodie", "lifestyle_base", 1024, 1024)
    assert m.shadow_no_strings is not None
    x, y = (int(v) for v in m.to_px(1.45, 7.0))
    assert m.shadow.crop(x, y, 1, 1).avg() < m.shadow_no_strings.crop(x, y, 1, 1).avg() - 20
    far = (int(m.to_px(4.5, 7.0)[0]), y)
    assert m.shadow.crop(*far, 1, 1).avg() == m.shadow_no_strings.crop(*far, 1, 1).avg()
    assert template_maps("tee", "lifestyle_base", 1024, 1024).shadow_no_strings is None
