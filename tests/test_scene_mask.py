"""The scene edit mask keeps the whole garment and figure; only the background is drawn (T-38-1).

Part 1 checks the mask PNG pixel by pixel against the template coverage. Part 2 runs the real
`composite_scene` on synthetic providers: one that obeys the mask must pass both checks, ones
that don't must fail. The checks themselves are untouched by this card.
"""

import numpy as np
import pytest
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

WHITE, BLACK, NAVY = "#F2F2F2", "#111111", "#1B3A6B"


def alpha_of(res) -> np.ndarray:
    return pyvips.Image.new_from_buffer(res.mask_png, "")[3].numpy()


def dilate(cover: np.ndarray, r: int) -> np.ndarray:
    """Boolean disc dilation by shifting: an implementation independent of the one under test."""
    out = np.zeros_like(cover)
    h, w = cover.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            ys, yd = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
            xs, xd = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
            out[yd, xd] |= cover[ys, xs]
    return out


# --- part 1: the mask -----------------------------------------------------------------------


@pytest.mark.parametrize("garment", list(SPECS))
@pytest.mark.parametrize("view", SCENE_VIEWS)
@pytest.mark.parametrize("size", [(1024, 1024), (1536, 1024)])
def test_mask_keeps_garment_and_figure_and_frees_the_background(garment, view, size):
    w, h = size
    res = scenes.render_base(garment, view, w, h, WHITE)
    maps = template_maps(garment, view, w, h)
    alpha = alpha_of(res)
    assert alpha.shape == (h, w) and set(np.unique(alpha)) <= {0, 255}

    cover = maps.product.numpy() > 0
    assert cover.any()
    assert (alpha[cover] == 255).all(), "every garment/figure pixel is kept"

    r = round(scenes.MASK_SILHOUETTE_MARGIN_IN * maps.ppi)
    assert (alpha[dilate(cover, r)] == 255).all(), "the rim is kept"

    px, py, pw, ph = res.protected_box_px
    box = np.zeros_like(cover)
    box[py : py + ph, px : px + pw] = True
    assert (alpha[box] == 255).all(), "print box + MASK_MARGIN_IN stays kept"

    far = ~dilate(cover, r + 2) & ~box
    assert far.any() and (alpha[far] == 0).all(), "the background is drawable"
    assert (alpha == 0).mean() > 0.25, "the provider has background to draw"
    # the rim is a disc, not a blur: no kept pixel beyond rim + 1 px (+ the print box)
    assert not (alpha[~dilate(cover, r + 1) & ~box] == 255).any()


def test_rim_default_is_none_and_stays_inside_half_the_outline_band():
    assert scenes.MASK_SILHOUETTE_MARGIN_IN < scenes.BAND_IN / 2
    assert scenes.MASK_MARGIN_IN == 0.5  # outline_check's ring and the restored patch use it
    for garment in SPECS:
        for view in SCENE_VIEWS:
            ppi = template_maps(garment, view, 1024, 1024).ppi
            assert round(scenes.MASK_SILHOUETTE_MARGIN_IN * ppi) == 0  # rim width in px at 1024


@pytest.mark.parametrize("view", SCENE_VIEWS)
def test_a_wider_rim_is_a_true_disc_to_within_a_pixel(monkeypatch, view):
    monkeypatch.setattr(scenes, "MASK_SILHOUETTE_MARGIN_IN", 0.2)
    res = scenes.render_base("hoodie", view, 1024, 1024, WHITE)
    maps = template_maps("hoodie", view, 1024, 1024)
    alpha = alpha_of(res)
    cover = maps.product.numpy() > 0
    r = round(0.2 * maps.ppi)
    assert r >= 3
    px, py, pw, ph = res.protected_box_px
    box = np.zeros_like(cover)
    box[py : py + ph, px : px + pw] = True
    assert (alpha[dilate(cover, r - 1)] == 255).all()
    assert (alpha[dilate(cover, r)] == 255).all()
    assert (alpha[~dilate(cover, r + 1) & ~box] == 0).all()


@pytest.mark.parametrize("view", SCENE_VIEWS)
def test_base_and_mask_still_carry_no_design_and_keep_the_response_shape(view):
    res = scenes.render_base("tee", view, 1024, 1024, NAVY)
    again = scenes.render_base("tee", view, 1024, 1024, NAVY)
    assert res.png == again.png and res.mask_png == again.mask_png  # deterministic
    maps = template_maps("tee", view, 1024, 1024)
    base = pyvips.Image.new_from_buffer(res.png, "")
    assert base(*maps.flat_px)[:3] == [0x1B, 0x3A, 0x6B]
    assert res.protected_box_px[2:] != (0, 0) and res.print_box_px == maps.print_area_px
    # the mask is colorless: RGB zero, alpha only
    mask = pyvips.Image.new_from_buffer(res.mask_png, "")
    assert mask[0:3].max() == 0
    # no version tag appended to the base's text chunk (read_base_meta splits it on "/")
    assert b"tee/" + view.encode() in res.png
    assert res.png.count(b"invai:scene-base") == 1


# --- part 2: providers through the real composite --------------------------------------------

S = 640  # smaller than the 1024 frames of the sim script; the checks are resolution-independent


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    path = tmp_path_factory.mktemp("maskdesign") / "design.png"
    im = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 560, 560), fill=(230, 60, 40, 255))
    d.rectangle((200, 150, 400, 450), fill=(250, 240, 60, 255))
    d.rectangle((260, 470, 340, 500), fill=(20, 20, 120, 255))
    im.save(path)
    return path


def run(tmp_path, design, res, scene, garment, view, blank):
    base_path, scene_path = tmp_path / "base.png", tmp_path / "scene.png"
    base_path.write_bytes(res.png)
    scene.pngsave(str(scene_path))
    return scenes.composite_scene(
        scene_path,
        base_path,
        res.print_box_px,
        design,
        6,
        6,
        blank,
        "shopify",
        "jpeg",
        [],
        garment=garment,
        view=view,
    )


@pytest.mark.parametrize("garment,view", [("tee", "lifestyle_base"), ("hoodie", "on_model_white")])
@pytest.mark.parametrize("blank", [BLACK, NAVY])
def test_a_provider_that_obeys_the_mask_passes_both_checks(tmp_path, design, garment, view, blank):
    res = scenes.render_base(garment, view, S, S, blank)
    scene = obeys_mask(res.png, res.mask_png, gain=0.92)
    r = run(tmp_path, design, res, scene, garment, view, blank)
    assert r.passes, (r.failures, r.region_reasons, r.region_unchanged_score)
    assert r.region_unchanged_score >= scenes.REGION_UNCHANGED_MIN
    assert r.design_lock_score >= scenes.DESIGN_LOCK_MIN and r.data


@pytest.mark.parametrize("view", SCENE_VIEWS)
def test_a_white_garment_passes_in_a_darker_room(tmp_path, design, view):
    res = scenes.render_base("tee", view, S, S, WHITE)
    room = busy_scene(S, S, tone=0.5)
    scene = obeys_mask(res.png, res.mask_png, 0.92, room)
    r = run(tmp_path, design, res, scene, "tee", view, WHITE)
    assert r.passes, (r.failures, r.region_reasons, r.region_unchanged_score)


@pytest.mark.parametrize("provider", [(S, S), (S * 3 // 2, S), (S, S * 3 // 2)])
def test_an_obedient_provider_that_rerenders_the_whole_frame_still_passes(
    tmp_path, design, provider
):
    """Kept pixels come back with noise, a slight blur and a JPEG round trip, at each size."""
    res = scenes.render_base("tee", "lifestyle_base", S, S, BLACK)
    frame = rerendered(obeys_mask(res.png, res.mask_png, gain=0.92))
    r = run(
        tmp_path, design, res, to_provider_size(frame, provider), "tee", "lifestyle_base", BLACK
    )
    assert r.passes, (provider, r.failures, r.region_reasons, r.region_unchanged_score)


@pytest.mark.parametrize("garment,view", [("tee", "lifestyle_base"), ("hoodie", "on_model_white")])
@pytest.mark.parametrize("blank", [BLACK, WHITE])
def test_a_provider_that_obeyed_only_the_old_mask_fails_check_1(
    tmp_path, design, garment, view, blank
):
    """The garment outside print box + 0.5 in widened 10% over a busy scene: the case the old
    mask allowed. Same checks, so the mask is what keeps a provider from doing this."""
    res = scenes.render_base(garment, view, S, S, blank)
    scene = old_mask_only(res.png, res.protected_box_px, garment, view, blank, widen=1.1)
    r = run(tmp_path, design, res, scene, garment, view, blank)
    assert not r.passes and r.failures == ["region_changed"], (r.failures, r.region_reasons)
    assert r.data is None and r.design_lock_score is None


def test_the_new_mask_makes_that_redrawn_region_a_kept_region():
    """Why (b) can't happen to a provider that obeys: the widened garment's pixels are opaque."""
    res = scenes.render_base("tee", "on_model_white", S, S, BLACK)
    maps = template_maps("tee", "on_model_white", S, S)
    alpha = alpha_of(res)
    assert (alpha[maps.product.numpy() > 0] == 255).all()
    old = np.zeros(alpha.shape, bool)
    px, py, pw, ph = res.protected_box_px
    old[py : py + ph, px : px + pw] = True
    assert (maps.product.numpy() > 0)[~old].any()  # garment outside the old protected box


def test_a_provider_that_redraws_the_garment_entirely_fails_check_1(tmp_path, design):
    res = scenes.render_base("tee", "lifestyle_base", S, S, BLACK)
    other = scenes.render_base("hoodie", "lifestyle_base", S, S, BLACK)
    frame = pyvips.Image.new_from_buffer(other.png, "")[0:3].copy(interpretation="srgb")
    r = run(tmp_path, design, res, frame, "tee", "lifestyle_base", BLACK)
    assert not r.passes and r.failures == ["region_changed"] and r.structural
    assert r.data is None


@pytest.mark.parametrize("blank,ink", [(BLACK, (230, 230, 230)), (NAVY, (240, 240, 240))])
def test_marks_drawn_in_the_print_area_fail_check_1b_and_write_nothing(
    tmp_path, design, blank, ink
):
    res = scenes.render_base("tee", "lifestyle_base", S, S, blank)
    scene = drawn_marks(obeys_mask(res.png, res.mask_png, gain=0.92), res.print_box_px, ink)
    r = run(tmp_path, design, res, scene, "tee", "lifestyle_base", blank)
    assert not r.passes and "box_foreign" in r.region_reasons
    assert r.failures == ["region_changed"] and r.data is None and r.design_lock_score is None
