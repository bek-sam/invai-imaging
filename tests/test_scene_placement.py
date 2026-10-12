"""Check 1c: garment size and place after registration (T-40-2, ADR 0039, B-374).

Check 1's outline score has little room on a narrowed garment (a sport-gray tee narrowed 10% on a
235 backdrop scores 0.689 against 0.75), and a garment-only move or scale is not a failure it can
see at all: registration is built to absorb a rigid shift and 0.90-1.10 scale of the whole frame,
and on `lifestyle_base` the garment is the whole product, so the print follows it through
registration and the output is right. The guard measures what registration cannot absorb: the
garment's extent (width, height) and a residual offset, in inches, from the outline of the
registered frame. Numbers come from `scripts/scene_guard_sweep.py`."""

import pytest
import pyvips
from scene_helpers import (
    flat_backdrop,
    obeys_mask,
    old_mask_only,
    perturbed,
    rerendered,
)

from app import scenes
from app.garments import template_maps

SIZE = 1024
VIEW = "lifestyle_base"
WHITE, BLACK, SPORT = "#F2F2F2", "#111111", "#9AA0A6"


class Case:
    def __init__(self, garment: str, blank: str) -> None:
        self.garment, self.blank = garment, blank
        self.res = scenes.render_base(garment, VIEW, SIZE, SIZE, blank)
        self.base = pyvips.Image.new_from_buffer(self.res.png, "")[0:3].copy_memory()
        self.maps = template_maps(garment, VIEW, SIZE, SIZE)

    def obedient(self, tone: int) -> pyvips.Image:
        gray = flat_backdrop(SIZE, SIZE, tone)
        return obeys_mask(self.res.png, self.res.mask_png, 0.92, gray)

    def redrawn(self, tone: int, **kw) -> pyvips.Image:
        gray = flat_backdrop(SIZE, SIZE, tone)
        return old_mask_only(
            self.res.png,
            self.res.protected_box_px,
            self.garment,
            VIEW,
            self.blank,
            background=gray,
            **kw,
        )

    def check(self, scene: pyvips.Image):
        """(reasons as composite_scene applies them, placement, check-1 score, registration)."""
        scene = scene.copy_memory()
        reg = scenes.register(scene, self.base, self.maps)
        c1 = scenes.outline_check(scene, self.base, self.maps, reg).score
        foreign = scenes.print_box_foreign(scene, self.base, self.maps, reg)
        pl = scenes.placement_check(scene, self.base, self.maps, reg)
        return scenes.region_reasons(c1, reg.at_limit, foreign, pl), pl, c1, reg


@pytest.fixture(scope="module")
def white_tee() -> Case:
    return Case("tee", WHITE)


@pytest.fixture(scope="module")
def black_tee() -> Case:
    return Case("tee", BLACK)


@pytest.fixture(scope="module")
def white_hoodie() -> Case:
    return Case("hoodie", WHITE)


# --- known good: no placement failure, whatever the frame does as a whole ----------------------


@pytest.mark.parametrize("case", ["white_tee", "black_tee"])
@pytest.mark.parametrize("tone", [105, 235])
def test_unchanged_and_rerendered_frames_have_no_offset(case, tone, request):
    c: Case = request.getfixturevalue(case)
    for scene in (c.obedient(tone), rerendered(c.obedient(tone))):
        reasons, pl, _, _ = c.check(scene)
        assert pl.sides == 4
        assert pl.size_in < 0.1 and pl.offset_in < 0.05, pl
        assert "placement_off" not in reasons


@pytest.mark.parametrize(
    "scale,shift,out",
    [
        (1.02, (7, -5), (1536, 1024)),  # mock-like
        (0.92, (0, 0), (1024, 1024)),
        (1.08, (0, 0), (1536, 1024)),
        (1.0, (40, 0), (1024, 1024)),  # 4% of the frame
        (0.95, (-30, 30), (1536, 1024)),
    ],
)
def test_whole_frame_shift_and_scale_are_absorbed_by_registration(white_tee, scale, shift, out):
    scene = perturbed(white_tee.base, "tee", VIEW, out, scale=scale, shift=shift)
    reasons, pl, _, reg = white_tee.check(scene)
    assert reg.scale == pytest.approx(scale, abs=0.03)
    assert pl.size_in < 0.3 and pl.offset_in < 0.1, pl
    assert "placement_off" not in reasons


@pytest.mark.parametrize(
    "kw",
    [
        {"dx_in": 1.0},
        {"dx_in": -1.0},
        {"shift_in": 1.0},
        {"shift_in": -1.0},
        {"scale": 0.95},
        {"scale": 1.05},
    ],
)
def test_garment_only_rigid_move_and_scale_read_as_a_frame_change(black_tee, kw):
    """On `lifestyle_base` a garment moved 1 in or scaled 5% against its own print box is the
    same thing as a whole-frame shift or scale. On a dark blank registration takes all of it
    (translation of 15 px = 1.0 in, scale 0.95/1.05), check 1 passes and so does the guard: the
    composite puts the print on the garment where the base has it. Not detectable and not
    harmful; pinned so nobody tunes the guard on it (on a white blank the kept print-box
    rectangle pulls registration back and check 1 fails these anyway)."""
    reasons, pl, c1, reg = black_tee.check(black_tee.redrawn(235, **kw))
    assert reasons == [] and c1 > 0.95
    assert pl.size_in < 0.1 and pl.offset_in < 0.05, pl
    assert max(abs(reg.tx), abs(reg.ty)) > 10 or abs(reg.scale - 1) > 0.04  # it really moved


def test_missing_outline_counts_zero_not_noise(white_tee):
    """No garment in the frame (a flat backdrop): nothing to measure on any side. Check 1 fails it
    (outline_low); the guard stays silent instead of adding a made-up offset."""
    scene = flat_backdrop(SIZE, SIZE, 200).cast("uchar").copy(interpretation="srgb")
    reasons, pl, _, _ = white_tee.check(scene)
    assert pl.sides == 0 and pl.size_in == 0 and pl.offset_in == 0
    assert "outline_low" in reasons and "placement_off" not in reasons


# --- known bad: reshaped garments --------------------------------------------------------------


@pytest.mark.parametrize("widen", [1.1, 0.9, 0.85])
@pytest.mark.parametrize("tone", [105, 200, 235])
@pytest.mark.parametrize("case", ["white_tee", "black_tee"])
def test_narrowed_or_widened_garment_is_placement_off_with_room(case, tone, widen, request):
    c: Case = request.getfixturevalue(case)
    reasons, pl, _, _ = c.check(c.redrawn(tone, widen=widen))
    assert "placement_off" in reasons
    assert pl.size_in > scenes.PLACEMENT_SIZE_MAX_IN * 1.25, (widen, tone, pl)  # 25% of room


@pytest.mark.parametrize("case", ["white_tee", "black_tee"])
def test_narrowed_5_percent_fails_check_1_one_way_or_the_other(case, request):
    """A 5% narrowing is the smallest redraw in the sweep and the guard alone does not always
    separate it (size 0.36-1.17 in against 0.5): check 1 itself fails every one of these rows
    (c1 <= 0.62), and either reason is enough."""
    c: Case = request.getfixturevalue(case)
    for tone in (105, 200, 235):
        reasons, _, _, _ = c.check(c.redrawn(tone, widen=0.95))
        assert reasons


def test_moved_garment_that_check_1_passes_is_caught_by_residual_offset(white_hoodie):
    """Red on 82e9b04: a hoodie moved up 0.3 in against its box on a 200 backdrop scores check 1 =
    1.000 with no reason (registration takes most of the move); 0.2 in of it is left, 0.05 in over
    the 0.15 in threshold."""
    reasons, pl, c1, reg = white_hoodie.check(white_hoodie.redrawn(200, shift_in=-0.3))
    assert c1 >= scenes.REGION_UNCHANGED_MIN and not reg.at_limit
    assert reasons == ["placement_off"]
    assert pl.offset_in > scenes.PLACEMENT_SHIFT_MAX_IN * 1.25


# --- reasons, structure, API -------------------------------------------------------------------


def test_region_reasons_adds_one_reason_and_keeps_the_others():
    pl = scenes.Placement(width_in=-1.0, sides=4)
    assert scenes.region_reasons(0.5, True, 0.1) == [
        "outline_low",
        "registration_at_limit",
        "box_foreign",
    ]
    assert scenes.region_reasons(0.5, True, 0.1, pl) == [
        "outline_low",
        "registration_at_limit",
        "box_foreign",
        "placement_off",
    ]
    assert scenes.region_reasons(0.9, False, 0.0, pl) == ["placement_off"]
    assert scenes.region_reasons(0.9, False, 0.0, scenes.Placement(width_in=0.4, sides=4)) == []
    assert scenes.region_reasons(0.9, False, 0.0, scenes.Placement(shift_x_in=0.14, sides=4)) == []
    assert scenes.region_reasons(0.9, False, 0.0, scenes.Placement(shift_y_in=-0.2, sides=4)) == [
        "placement_off"
    ]


def test_placement_only_failure_is_structural_only_when_registration_cannot_line_up():
    """ADR 0039: `is_structural` is unchanged. Registration below 0.5 means no scale and shift
    matches the base's garment, whatever the reason; a placement-only failure with a good
    registration is this draw's width and a regenerate can land closer, so it retries."""
    assert scenes.is_structural(["placement_off"], 0.45) is True
    assert scenes.is_structural(["placement_off"], 0.8) is False
    assert scenes.is_structural([], 0.1) is False


def test_composite_reports_placement_off_and_saves_nothing(tmp_path, white_hoodie):
    """The API shape is unchanged: the reason is one more string in `region_reasons`, the failure
    stays `region_changed`, and no composite is returned."""
    c = white_hoodie
    scene, base, design = tmp_path / "scene.png", tmp_path / "base.png", tmp_path / "d.png"
    c.redrawn(200, shift_in=-0.3).pngsave(str(scene))
    base.write_bytes(c.res.png)
    pyvips.Image.black(300, 300, bands=4).draw_rect(
        [200, 30, 30, 255], 20, 20, 260, 260, fill=True
    ).copy(interpretation="srgb").pngsave(str(design))
    res = scenes.composite_scene(
        scene, base, c.maps.print_area_px, design, 6, 6, WHITE, "etsy", "jpeg", [],
        garment="hoodie", view=VIEW,
    )  # fmt: skip
    assert res.data is None and not res.passes
    assert res.failures == ["region_changed"]
    assert res.region_reasons == ["placement_off"]
    assert res.region_unchanged_score >= scenes.REGION_UNCHANGED_MIN
    assert res.design_lock_score is None


def test_check_2_result_is_unchanged_on_a_clean_scene(tmp_path, black_tee):
    """Check 2 is not touched by T-40-2: this obedient black-tee frame scored 0.9843 on 82e9b04 and
    scores it now (the corpus frames' check-2 values are identical before and after, see the T-40-2
    report)."""
    c = black_tee
    scene, base, design = tmp_path / "scene.png", tmp_path / "base.png", tmp_path / "d.png"
    c.obedient(235).pngsave(str(scene))
    base.write_bytes(c.res.png)
    pyvips.Image.black(300, 300, bands=4).draw_rect(
        [200, 30, 30, 255], 20, 20, 260, 260, fill=True
    ).copy(interpretation="srgb").pngsave(str(design))
    res = scenes.composite_scene(
        scene, base, c.maps.print_area_px, design, 6, 6, BLACK, "etsy", "jpeg", [],
        garment="tee", view=VIEW,
    )  # fmt: skip
    assert res.passes and res.region_reasons == []
    assert res.design_lock_score == pytest.approx(PINNED_CHECK_2, abs=1e-4)


PINNED_CHECK_2 = 0.9843  # measured on 82e9b04 (design_lock_score is rounded to 4 places)
