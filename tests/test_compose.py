import pytest
import pyvips
import zxingcpp
from PIL import Image
from pypdf import PdfReader

from app.compose import (
    CUT_GUIDE_COLOR,
    ComposePlacement,
    Label,
    _qr_matrix,
    compose_sheet,
    label_text,
    qr_min_side_in,
)
from app.storage import LocalStorage
from app.vips import ImageError
from tests.helpers import solid_design

DPI = 300
# Content height (label_height_in - label_gap_in) tall enough for a UUID transfer_id's QR to
# clear the AC1 0.5mm/module floor at the default label_gap_in (0.125in): see
# test_qr_module_floor_is_dpi_independent for the exact boundary this margin sits above.
LABEL_HEIGHT_IN = 0.8
GAP_IN = 0.125


def test_compose_small_sheet(tmp_path):
    store = LocalStorage(tmp_path / "store")
    solid_design(tmp_path / "store" / "a.png", 600, 900)
    placements = [
        ComposePlacement(
            "11111111-2222-3333-4444-555555555555",
            "a.png",
            0.25,
            0.25,
            2.0,
            3.0,
            False,
            Label("DB-1001", "1/2", "M", "Black", "Desert Bloom"),
        ),
        # rotated copy: the 2x3 art placed as 3x2
        ComposePlacement(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "a.png",
            2.5,
            0.25,
            3.0,
            2.0,
            True,
            Label("DB-1002", "2/2", "L", "White", "Desert Bloom", reprint=True),
        ),
    ]
    res = compose_sheet(
        store,
        6.0,
        4.5,
        DPI,
        placements,
        LABEL_HEIGHT_IN,
        "out/sheet.png",
        "out/sheet.pdf",
        "out/preview.png",
        300,
        label_gap_in=GAP_IN,
    )
    assert (res.width_px, res.height_px) == (1800, 1350)

    sheet = pyvips.Image.new_from_file(str(tmp_path / "store/out/sheet.png"))
    assert (sheet.width, sheet.height, sheet.bands) == (1800, 1350, 4)
    assert round(sheet.xres * 25.4) == DPI
    # background stays transparent, the design is opaque
    assert sheet.getpoint(1790, 1340)[3] == 0
    assert sheet.getpoint(round(1.25 * DPI), round(1.75 * DPI))[3] == 255
    # rotated art: the red top band of the design ends up on the right edge (rot90 clockwise)
    top_right = sheet.getpoint(round(5.4 * DPI), round(1.25 * DPI))
    assert top_right[0] > 200 and top_right[2] < 100

    # the thin cut guide sits in the gap, between the first design and its label
    gap_px = round(GAP_IN * DPI)
    guide_h = max(1, round(0.006 * DPI))
    design1_bottom = round(0.25 * DPI) + round(3.0 * DPI)
    guide_y = design1_bottom + (gap_px - guide_h) // 2
    guide_x = round(0.25 * DPI) + 2  # inside the first dash, which starts at the design's x
    assert sheet.getpoint(guide_x, guide_y) == list(CUT_GUIDE_COLOR)

    # QR codes in the label strips decode to the transfer ids
    pil = Image.open(tmp_path / "store/out/sheet.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert found == {p.transfer_id for p in placements}

    preview = pyvips.Image.new_from_file(str(tmp_path / "store/out/preview.png"))
    assert (preview.width, preview.height) == (300, 225)

    page = PdfReader(tmp_path / "store/out/sheet.pdf").pages[0]
    assert float(page.mediabox.width) / 72 == 6.0
    assert float(page.mediabox.height) / 72 == 4.5
    img = page["/Resources"]["/XObject"]["/Im0"]
    assert (img["/Width"], img["/Height"]) == (1800, 1350)


# B-80: PDF pages over 200in no longer use /UserUnit (see tests/test_pdf.py) — capped instead.


def test_label_text():
    assert label_text(Label("A1", "2", "M", "", "Cactus")) == "A1 · 2 · M · Cactus"


# --- AC1: transfer-label QR module floor -----------------------------------------------------


def test_qr_min_side_in_matches_matrix_size():
    from app.compose import MIN_MODULE_MM
    from app.vips import MM_PER_IN

    n = len(_qr_matrix("11111111-2222-3333-4444-555555555555"))
    assert qr_min_side_in("11111111-2222-3333-4444-555555555555") == n * MIN_MODULE_MM / MM_PER_IN


def test_qr_module_floor_is_dpi_independent():
    """The 0.5mm/module floor is pure geometry (module count x mm), not a function of DPI."""
    tid = "11111111-2222-3333-4444-555555555555"
    min_side_in = qr_min_side_in(tid)
    assert min_side_in > 0.5  # a UUID needs more than a single 0.5mm module, obviously
    for dpi in (150, 300, 600):
        from app.compose import qr_image

        # just under the boundary at every DPI: fails loudly, whatever the sheet's own DPI
        with pytest.raises(ImageError, match="scannable"):
            qr_image(tid, round((min_side_in - 0.01) * dpi), dpi)
        # just over it: renders fine
        qr_image(tid, round((min_side_in + 0.01) * dpi), dpi)


def test_compose_fails_loudly_when_label_strip_too_short(tmp_path):
    store = LocalStorage(tmp_path / "store")
    solid_design(tmp_path / "store" / "a.png", 600, 900)
    tid = "11111111-2222-3333-4444-555555555555"
    placements = [
        ComposePlacement(tid, "a.png", 0.25, 0.25, 2.0, 3.0, False, None),
    ]
    min_side_in = qr_min_side_in(tid)

    # Just below the label-strip height that would give a 0.5mm-module QR: ImageError, not a
    # silently-shrunk, unscannable code.
    too_short = min_side_in + GAP_IN - 0.02
    with pytest.raises(ImageError, match="scannable"):
        compose_sheet(
            store,
            6.0,
            5.0,
            DPI,
            placements,
            too_short,
            "fail/o.png",
            None,
            "fail/p.png",
            100,
            label_gap_in=GAP_IN,
        )

    # Just above it: succeeds and the QR decodes.
    ok_height = min_side_in + GAP_IN + 0.02
    compose_sheet(
        store,
        6.0,
        5.0,
        DPI,
        placements,
        ok_height,
        "ok/o.png",
        None,
        "ok/p.png",
        100,
        label_gap_in=GAP_IN,
    )
    pil = Image.open(tmp_path / "store/ok/o.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert found == {tid}


@pytest.mark.parametrize("dpi", [150, 200, 300])
def test_compose_qr_decodes_at_multiple_dpi(tmp_path, dpi):
    store = LocalStorage(tmp_path / "store" / str(dpi))
    solid_design(tmp_path / "store" / str(dpi) / "a.png", 600, 900)
    tid = "3f9a1c22-56e2-4b7a-9f21-abcdef012345"
    placements = [
        ComposePlacement(
            tid, "a.png", 0.25, 0.25, 2.0, 3.0, False, Label("DB-1", "1/1", "M", "Black", "Cactus")
        )
    ]
    compose_sheet(
        store,
        3.0,
        4.5,
        dpi,
        placements,
        LABEL_HEIGHT_IN,
        "o.png",
        None,
        "p.png",
        100,
        label_gap_in=GAP_IN,
    )
    pil = Image.open(tmp_path / "store" / str(dpi) / "o.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert found == {tid}


# --- AC2: sheet header ------------------------------------------------------------------------


def test_compose_header_decodes(tmp_path):
    store = LocalStorage(tmp_path / "store")
    solid_design(tmp_path / "store" / "a.png", 600, 900)
    tid = "11111111-2222-3333-4444-555555555555"
    filename_hint = "SH-1042_DB-1001"
    # Enough top clearance (y_in=1.0) for the header QR to clear its own module floor.
    placements = [
        ComposePlacement(
            tid,
            "a.png",
            0.25,
            1.0,
            2.0,
            3.0,
            False,
            Label("DB-1001", "1/1", "M", "Black", "Cactus"),
        )
    ]
    compose_sheet(
        store,
        6.0,
        5.5,
        DPI,
        placements,
        LABEL_HEIGHT_IN,
        "h/o.png",
        None,
        "h/p.png",
        100,
        label_gap_in=GAP_IN,
        filename_hint=filename_hint,
    )
    pil = Image.open(tmp_path / "store/h/o.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert filename_hint in found
    assert tid in found


def test_compose_without_filename_hint_has_no_header(tmp_path):
    store = LocalStorage(tmp_path / "store")
    solid_design(tmp_path / "store" / "a.png", 600, 900)
    tid = "11111111-2222-3333-4444-555555555555"
    placements = [ComposePlacement(tid, "a.png", 0.25, 1.0, 2.0, 3.0, False, None)]
    compose_sheet(
        store,
        6.0,
        5.5,
        DPI,
        placements,
        LABEL_HEIGHT_IN,
        "n/o.png",
        None,
        "n/p.png",
        100,
        label_gap_in=GAP_IN,
    )
    sheet = pyvips.Image.new_from_file(str(tmp_path / "store/n/o.png"))
    # the top margin band stays fully transparent -- no header was drawn
    assert sheet.getpoint(10, 10)[3] == 0
