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
# A generous label_height_in for tests that aren't specifically about the AC1 module floor (see
# DEFAULT_LABEL_HEIGHT_IN below and test_default_label_height_decodes_a_real_uuid for that).
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
    """The 0.33mm/module floor is pure geometry (module count x mm), not a function of DPI."""
    from app.compose import MIN_MODULE_MM

    tid = "11111111-2222-3333-4444-555555555555"
    min_side_in = qr_min_side_in(tid)
    assert min_side_in > MIN_MODULE_MM / 25.4  # needs more than a single module, obviously
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
    # label_gap_in only positions the label below the design now (AC3); it no longer shrinks the
    # label's own content height (AC1 needs every bit of label_height_in), so the boundary is
    # qr_min_side_in itself, not label_height_in - label_gap_in.
    min_side_in = qr_min_side_in(tid)

    # Just below the label-strip height that would give a 0.33mm-module QR: ImageError, not a
    # silently-shrunk, unscannable code.
    too_short = min_side_in - 0.02
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
    ok_height = min_side_in + 0.02
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


DEFAULT_LABEL_HEIGHT_IN = 0.42  # must match NestRequest/ComposeRequest's Field default in main.py


@pytest.mark.parametrize("dpi", [150, 200, 300])
def test_default_label_height_decodes_a_real_uuid(tmp_path, dpi):
    """Tech lead follow-up to the T-9-2 report's "AC1 finding": raising the shared
    label_height_in default (imaging NestRequest/ComposeRequest, backend LABEL_HEIGHT_IN) from
    0.35in to 0.42in closes the 0.053in gap a real (random) UUID transfer_id's QR (31 modules
    with the quiet zone kept, 0.403in needed at the 0.33mm floor -- geometry, not raster) was
    short by. Proves it at the *default* height, with a fresh random UUID each run, decoded at
    150/200/300 DPI -- not the taller LABEL_HEIGHT_IN this file's other fixtures use.
    """
    import uuid

    store = LocalStorage(tmp_path / "store" / str(dpi))
    solid_design(tmp_path / "store" / str(dpi) / "a.png", 600, 900)
    tid = str(uuid.uuid4())
    placements = [
        ComposePlacement(
            tid, "a.png", 0.25, 0.25, 2.0, 3.0, False, Label("DB-1", "1/1", "M", "Black", "Cactus")
        )
    ]
    compose_sheet(
        store, 6.0, 4.0, dpi, placements, DEFAULT_LABEL_HEIGHT_IN, "o.png", None, "p.png", 100
    )
    pil = Image.open(tmp_path / "store" / str(dpi) / "o.png").convert("RGB")
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
    # header_height_in is a dedicated top-edge band now (tech lead, 2026-09-26): it no longer
    # depends on the placement's y_in, just kept generous here so nothing overlaps the header.
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


@pytest.mark.parametrize("dpi", [150, 200, 300])
def test_default_settings_header_and_transfer_qrs_all_decode(tmp_path, dpi):
    """Tech lead follow-up: a dedicated top header_height_in band (default 0.45in, nesting
    starts below it) instead of growing margin_in on every side. Proves the whole pipeline at
    *default* settings -- real nesting.nest() placements, default label_height_in/label_gap_in/
    header_height_in in compose_sheet -- decodes both the header and every transfer QR, with
    fresh random UUIDs, at 150/200/300 DPI.
    """
    import uuid

    from app.nesting import Item
    from app.nesting import nest as nest_fn

    tids = [str(uuid.uuid4()) for _ in range(2)]
    sheets = nest_fn([Item(tid, 2.0, 3.0) for tid in tids])  # nesting.nest()'s own defaults
    sheet = sheets[0]

    store = LocalStorage(tmp_path / "store" / str(dpi))
    solid_design(tmp_path / "store" / str(dpi) / "a.png", 600, 900)
    filename_hint = "SH-DEFAULT DB-1 DB-2"
    placements = [
        ComposePlacement(
            p.id,
            "a.png",
            p.x_in,
            p.y_in,
            p.width_in,
            p.height_in,
            p.rotated,
            Label("DB-1", "1/1", "M", "Black", "Cactus"),
        )
        for p in sheet.placements
    ]
    compose_sheet(
        store,
        22.0,
        sheet.length_in,
        dpi,
        placements,
        0.42,  # matches nesting.nest()'s and ComposeRequest's default label_height_in
        "o.png",
        None,
        "p.png",
        100,
        filename_hint=filename_hint,  # label_gap_in and header_height_in: compose's own defaults
    )
    pil = Image.open(tmp_path / "store" / str(dpi) / "o.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert set(tids) <= found
    assert filename_hint in found
