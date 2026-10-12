"""Check 1 on a seamless studio backdrop (T-39-1): which frames the outline check must keep
failing, what it does with a garment we kept on an unlucky backdrop, and the diagnosis tool.

The wave 38 trial's `studio` scenes failed check 1 in 3 of 4 scenes (white tee 2 of 2). Offline
diagnosis (`scripts/scene_outline_diag.py`, README "Scene lock diagnostics"): the provider kept
the garment pixels in every failing frame; what failed is the outline, whose drawn stroke lost its
edge against a light gray backdrop 7-25 levels from it. A flat-gray sweep reproduces the dip with
no redraw at all (white tee: backdrop levels 176-200 fail, 206 and up and 170 and down pass). The
check was not changed: these tests pin both sides of that, so a future measure that recovers the
unlucky backdrops has to keep every bad frame below failing too."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pyvips
from scene_helpers import flat_backdrop, obeys_mask, old_mask_only, rerendered

from app import scenes
from app.fonts import parse_hex
from app.garments import template_maps

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPTS))
    return mod


diag = _load("scene_outline_diag")
sim = _load("scene_mask_sim")

SIZE = 1024
VIEW = "lifestyle_base"
WHITE, BLACK = "#F2F2F2", "#111111"


def luma_level(hexv: str) -> int:
    r, g, b = parse_hex(hexv)[:3]
    return round(0.2126 * r + 0.7152 * g + 0.0722 * b)


class Case:
    """A base, its mask and the check-1 score of any frame against it."""

    def __init__(self, garment: str, blank: str) -> None:
        self.garment, self.blank = garment, blank
        self.res = scenes.render_base(garment, VIEW, SIZE, SIZE, blank)
        self.base = pyvips.Image.new_from_buffer(self.res.png, "")[0:3].copy_memory()
        self.maps = template_maps(garment, VIEW, SIZE, SIZE)
        self.keep = pyvips.Image.new_from_buffer(self.res.mask_png, "")[3].numpy() == 255

    def c1(self, scene: pyvips.Image) -> float:
        reg = scenes.register(scene, self.base, self.maps)
        return scenes.outline_check(scene, self.base, self.maps, reg).score

    def obedient(self, tone: int, rerender: bool = False) -> pyvips.Image:
        frame = obeys_mask(self.res.png, self.res.mask_png, 0.92, flat_backdrop(SIZE, SIZE, tone))
        return rerendered(frame) if rerender else frame

    def old_mask(self, tone: int, **kw) -> pyvips.Image:
        return old_mask_only(
            self.res.png,
            self.res.protected_box_px,
            self.garment,
            VIEW,
            self.blank,
            background=flat_backdrop(SIZE, SIZE, tone),
            **kw,
        )


@pytest.fixture(scope="module")
def white_tee():
    return Case("tee", WHITE)


@pytest.fixture(scope="module")
def black_tee():
    return Case("tee", BLACK)


# --- the guarantee: bad frames keep failing, on every backdrop ----------------------------------


@pytest.mark.parametrize("tone", [250, 215, 128, 10])
def test_widened_narrowed_or_redrawn_garment_fails_on_a_flat_backdrop(white_tee, black_tee, tone):
    for case in (white_tee, black_tee):
        other = Case("hoodie", case.blank)
        omaps = other.maps
        redrawn = (
            omaps.product.ifthenelse(
                other.base.cast("float"), flat_backdrop(SIZE, SIZE, tone), blend=True
            )
            .cast("uchar")
            .copy(interpretation="srgb")
        )
        bad = {
            "widened 10%": case.old_mask(tone, widen=1.1),
            "narrowed 10%": case.old_mask(tone, widen=0.9),
            "another garment": redrawn,
            "garment absent": flat_backdrop(SIZE, SIZE, tone)
            .cast("uchar")
            .copy(interpretation="srgb"),
        }
        for name, scene in bad.items():
            c1 = case.c1(scene)
            assert c1 < scenes.REGION_UNCHANGED_MIN, f"{name} on {case.blank} at tone {tone}: {c1}"


@pytest.mark.parametrize("blank", [WHITE, BLACK])
def test_widened_garment_on_a_backdrop_the_color_of_the_blank_fails(blank):
    # Edges can't tell the widened garment from the backdrop here: it must fail, not pass.
    case = Case("tee", blank)
    scene = case.old_mask(luma_level(blank), widen=1.1)
    assert case.c1(scene) < scenes.REGION_UNCHANGED_MIN


def test_obedient_frames_still_pass_where_the_outline_stands_out(white_tee, black_tee):
    for tone in (250, 128):
        assert white_tee.c1(white_tee.obedient(tone, rerender=True)) >= scenes.REGION_UNCHANGED_MIN
    for tone in (250, 190):
        assert black_tee.c1(black_tee.obedient(tone, rerender=True)) >= scenes.REGION_UNCHANGED_MIN


# --- the diagnosis: a faint outline on a kept garment ------------------------------------------


def test_an_unlucky_flat_backdrop_fails_check_1_on_a_garment_we_kept(white_tee):
    """The measured limit (README): the same obedient frame passes on a near-white or mid-gray
    backdrop and fails in a window just under the garment's drawn outline. Fidelity says the
    garment is ours (not a redraw); the diagnosis must not call it redrawn. If check 1's measure
    ever changes to recover these, this test flips and the bad-frame tests above are the guard."""
    unlucky = white_tee.obedient(194, rerender=True)
    assert white_tee.c1(unlucky) < scenes.REGION_UNCHANGED_MIN
    for tone in (250, 212, 170):
        assert white_tee.c1(white_tee.obedient(tone, rerender=True)) >= scenes.REGION_UNCHANGED_MIN
    m = diag.measure(unlucky, white_tee.base, white_tee.maps, white_tee.keep, (242, 242, 242))
    assert m["mad"] <= diag.FID_MAD_GOOD and m["ssim"] >= diag.FID_SSIM_GOOD
    assert m["spill"] <= diag.SPILL_MAX
    assert diag.verdict(m, 0.949) in ("ours, low contrast", "undecidable")


def test_diag_calls_another_garment_redrawn_and_an_obedient_frame_good(white_tee):
    omaps = template_maps("hoodie", VIEW, SIZE, SIZE)
    ores = scenes.render_base("hoodie", VIEW, SIZE, SIZE, WHITE)
    oimg = pyvips.Image.new_from_buffer(ores.png, "")[0:3]
    gray = flat_backdrop(SIZE, SIZE, 128)
    other = omaps.product.ifthenelse(oimg.cast("float"), gray, blend=True).cast("uchar")
    m = diag.measure(
        other.copy(interpretation="srgb"),
        white_tee.base,
        white_tee.maps,
        white_tee.keep,
        (242,) * 3,
    )
    assert diag.verdict(m, 0.949) == "redrawn"
    good = diag.measure(
        white_tee.obedient(128, rerender=True),
        white_tee.base,
        white_tee.maps,
        white_tee.keep,
        (242,) * 3,
    )
    assert good["mad"] <= diag.FID_MAD_GOOD and good["ssim"] >= diag.FID_SSIM_GOOD
    assert good["edge_frame"] / good["edge_base"] > diag.CONTRAST_LOW_RATIO


def test_diag_fidelity_helpers_on_known_arrays():
    flat = np.full((64, 64), 100.0, dtype=np.float32)
    assert diag.ssim(flat, flat).min() == pytest.approx(1.0)
    disc = np.zeros((41, 41), dtype=bool)
    disc[10:31, 10:31] = True
    grown = diag.morph(disc, 3, "dilate")
    shrunk = diag.morph(disc, 3, "erode")
    assert grown.sum() > disc.sum() > shrunk.sum()
    assert shrunk[13:28, 13:28].all() and not shrunk[10, 10]


# --- the sweep tool ----------------------------------------------------------------------------


def test_blank_luma_bands_cover_the_whole_scale_and_match_the_sweep_blanks():
    for lum in np.linspace(0, 1, 101):
        sim.band_of(float(lum))  # no gap, no overlap that raises
    assert sim.band_of(0.949) == "white" and sim.band_of(0.067) == "black"
    assert sim.band_of(sim.blank_luma("#C41E3A")) == "mid"
    assert sim.band_of(sim.blank_luma("#1F2A44")) == "dark"
    assert sim.LIGHT_GRAY in sim.SWEEP_TONES and len(sim.SWEEP_TONES) >= 8


def test_sweep_summary_reports_best_tone_and_the_known_bad_gap(tmp_path, capsys):
    rows = ["\t".join(sim.TSV_COLS)]
    for tone, c1 in ((250, 1.0), (200, 0.6), (128, 0.9)):
        for case in ("a", "a'"):
            rows.append(f"{case}\ttee\tlifestyle_base\twhite\t0.949\t{tone}\t{c1}\t0.9")
    rows.append("widen10\ttee\tlifestyle_base\twhite\t0.949\t250\t0.2\t0.8")
    rows.append("shift0.6\ttee\tlifestyle_base\twhite\t0.949\t250\t0.9\t0.8")
    path = tmp_path / "rows.tsv"
    path.write_text("\n".join(rows) + "\n")
    assert sim.summarize([path]) == 0
    out = capsys.readouterr().out
    assert "best tone 250" in out and "every case passes at: [250, 128]" in out
    assert (
        "highest asserted bad score 0.200 -> gap to 0.75 is 0.550" in out
    )  # shift0.6 not asserted
