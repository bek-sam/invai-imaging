import pyvips
from PIL import Image as PILImage

from app.render import Slot, fit_lines, render_personalization, render_slot


def flags_for(res, slot):
    return {f.code for f in res.flags if f.slot == slot}


def test_shrinks_to_fit_then_flags_overflow(tmp_path):
    slots = [
        Slot("name", 0.5, 0.5, 4.5, 2, font_family="Oswald", font_size_pt=110),
        Slot("fits", 0.5, 3, 9, 2, font_size_pt=60),
        Slot("long", 0.5, 5.5, 3, 1, font_size_pt=72, max_chars=10),
        Slot("empty", 0.5, 7, 3, 1),
        Slot("odd", 0.5, 8, 9, 1, font_size_pt=24),
    ]
    values = {
        "name": "JACKSON",  # 5.6in wide at 110pt; fits the 4.5in slot at ~80%
        "fits": "Ava",
        "long": "Maximiliana Rodriguez",
        "odd": "Grandma 💖",
    }
    out = tmp_path / "p.png"
    res = render_personalization(10, 9.5, slots, values, 300, out)
    assert (res.width_px, res.height_px) == (3000, 2850)
    assert flags_for(res, "name") == set()
    assert flags_for(res, "fits") == set()
    assert flags_for(res, "long") == {"too_long", "overflow"}
    assert flags_for(res, "empty") == {"empty"}
    assert "suspicious_chars" in flags_for(res, "odd")

    img = pyvips.Image.new_from_file(str(out))
    assert (img.width, img.height, img.bands) == (3000, 2850, 4)
    # something was drawn inside the "name" slot, nothing in the empty one
    name = img.crop(150, 150, 1350, 600)[3]
    assert name.max() == 255
    # shrunk text stays inside its slot: nothing right of it on that row
    assert img.crop(1500, 150, 1500, 600)[3].max() == 0
    assert img.crop(150, 2100, 900, 300)[3].max() == 0


# --- AC1: multi-line, max-lines, min-font-size floor -----------------------------------------


def test_fit_lines_wraps_across_multiple_lines_when_it_fits():
    slot = Slot("bio", 0, 0, 3, 3, font_size_pt=40)
    lines, _f, fits = fit_lines(slot, "Happy Birthday Dear Friend", 150, 3 * 150, 3 * 150)
    assert fits is True
    assert len(lines) > 1
    assert " ".join(lines).split() == ["Happy", "Birthday", "Dear", "Friend"]


def test_fit_lines_flags_overflow_past_max_lines():
    slot = Slot("bio", 0, 0, 3, 1, font_size_pt=60, max_lines=1)
    lines, _f, fits = fit_lines(slot, "Happy Birthday Dear Friend", 150, 3 * 150, 1 * 150)
    assert fits is False
    assert len(lines) == 1  # capped to max_lines even though it doesn't all fit


def test_min_font_size_pt_wins_when_more_generous_than_min_scale(tmp_path):
    # MIN_SCALE alone would floor at 60% * 40pt = 24pt; min_font_size_pt=32pt is more generous
    # (larger), so per the wave decision the slot shrinks no further than 32pt.
    slot = Slot("x", 0, 0, 1, 0.3, font_size_pt=40, min_font_size_pt=32, max_lines=1)
    _lines, f, fits = fit_lines(
        slot, "averyveryverylongwordthatoverflows", 300, 1 * 300, round(0.3 * 300)
    )
    assert fits is False
    assert f.size == round(32 / 72 * 300)


def test_multiline_render_end_to_end_no_overflow(tmp_path):
    slot = Slot("bio", 0.2, 0.2, 3.6, 3.6, font_size_pt=48, max_lines=4)
    out = tmp_path / "bio.png"
    res = render_personalization(
        4, 4, [slot], {"bio": "Happy Birthday Dear Friend Forever"}, 150, out
    )
    assert flags_for(res, "bio") == set()


# --- AC2: outline (stroke) ---------------------------------------------------------------------


def test_outline_stroke_widens_the_glyph_ink():
    plain = Slot("t", 0, 0, 2, 1, font_size_pt=72, color="#000000")
    outlined = Slot(
        "t", 0, 0, 2, 1, font_size_pt=72, color="#000000", stroke_width_pt=8, stroke_color="#ffffff"
    )
    im_plain, fits_plain = render_slot(plain, "I", 150)
    im_outlined, fits_outlined = render_slot(outlined, "I", 150)
    assert fits_plain and fits_outlined
    bp, bo = im_plain.getbbox(), im_outlined.getbbox()
    assert (bo[2] - bo[0]) > (bp[2] - bp[0])
    assert (bo[3] - bo[1]) > (bp[3] - bp[1])


# --- AC3: photo slots ---------------------------------------------------------------------------


def test_photo_slot_fit_letterboxes_and_fill_covers(tmp_path):
    photo_path = tmp_path / "photo.png"
    PILImage.new("RGB", (1200, 300), (200, 30, 30)).save(
        photo_path
    )  # 4:1 aspect, ~150 DPI at 2x2in

    out_fit = tmp_path / "fit.png"
    res_fit = render_personalization(
        2,
        2,
        [Slot("p", 0, 0, 2, 2, kind="photo", fit="fit")],
        {},
        100,
        out_fit,
        photos={"p": photo_path},
    )
    # 4:1 photo in a 1:1 slot is a big aspect mismatch (B-103); the photo is downscaled, not
    # upscaled, so no "upscale" flag.
    assert flags_for(res_fit, "p") == {"aspect_mismatch"}
    img_fit = pyvips.Image.new_from_file(str(out_fit))
    # letterboxed to fit the width: the top corners of the square canvas stay transparent
    assert img_fit.crop(0, 0, 10, 10)[3].max() == 0

    out_fill = tmp_path / "fill.png"
    res_fill = render_personalization(
        2,
        2,
        [Slot("p", 0, 0, 2, 2, kind="photo", fit="fill")],
        {},
        100,
        out_fill,
        photos={"p": photo_path},
    )
    assert flags_for(res_fill, "p") == {"aspect_mismatch"}
    img_fill = pyvips.Image.new_from_file(str(out_fill))
    # filled to cover the box: every corner is opaque
    assert img_fill.crop(0, 0, 10, 10)[3].min() == 255


def test_low_res_photo_is_flagged(tmp_path):
    photo_path = tmp_path / "small.png"
    PILImage.new("RGB", (50, 50), (0, 0, 0)).save(photo_path)
    out = tmp_path / "out.png"
    # 4x4in slot from a 50x50px photo is ~12.5 effective DPI, far under the 150 DPI floor, and
    # needs an 8x upscale (400px / 50px) to fill it -- both get flagged.
    res = render_personalization(
        4, 4, [Slot("p", 0, 0, 4, 4, kind="photo")], {}, 100, out, photos={"p": photo_path}
    )
    assert flags_for(res, "p") == {"low_res_photo", "upscale"}


def test_photo_slot_without_a_value_is_flagged_empty(tmp_path):
    out = tmp_path / "out.png"
    res = render_personalization(
        2, 2, [Slot("p", 0, 0, 2, 2, kind="photo")], {}, 100, out, photos={}
    )
    assert flags_for(res, "p") == {"empty"}


def test_photo_slot_upscale_flagged_without_aspect_mismatch(tmp_path):
    photo_path = tmp_path / "small-square.png"
    PILImage.new("RGB", (350, 350), (10, 10, 10)).save(photo_path)  # same 1:1 ratio as the slot
    out = tmp_path / "out.png"
    # 2x2in slot at 300 DPI is 600x600px: a 1.71x upscale from the 350x350px photo, but the photo
    # is still 175 DPI at the slot's physical size (above the 150 floor) -- isolates the upscale
    # flag from low_res_photo.
    res = render_personalization(
        2, 2, [Slot("p", 0, 0, 2, 2, kind="photo")], {}, 300, out, photos={"p": photo_path}
    )
    assert flags_for(res, "p") == {"upscale"}


def test_photo_slot_aspect_mismatch_without_upscale(tmp_path):
    photo_path = tmp_path / "wide.png"
    PILImage.new("RGB", (900, 300), (10, 10, 10)).save(photo_path)  # 3:1, plenty of pixels
    out = tmp_path / "out.png"
    res = render_personalization(
        2,
        2,
        [Slot("p", 0, 0, 2, 2, kind="photo", fit="fill")],
        {},
        100,
        out,
        photos={"p": photo_path},
    )
    assert flags_for(res, "p") == {"aspect_mismatch"}


# --- AC4: missing glyphs -------------------------------------------------------------------------


def test_missing_glyphs_flag_lists_unsupported_characters(tmp_path):
    out = tmp_path / "out.png"
    res = render_personalization(
        3,
        1,
        [Slot("name", 0, 0, 3, 1, font_family="Bebas Neue", font_size_pt=40)],
        {"name": "Testɂ"},
        150,
        out,
    )
    flags = [f for f in res.flags if f.slot == "name" and f.code == "missing_glyphs"]
    assert len(flags) == 1
    assert "ɂ" in flags[0].message
