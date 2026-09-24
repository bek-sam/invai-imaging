import pyvips
import zxingcpp
from PIL import Image
from pypdf import PdfReader

from app.compose import ComposePlacement, Label, compose_sheet, label_text
from app.storage import LocalStorage
from tests.helpers import solid_design

DPI = 300


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
        4.0,
        DPI,
        placements,
        0.35,
        "out/sheet.png",
        "out/sheet.pdf",
        "out/preview.png",
        300,
    )
    assert (res.width_px, res.height_px) == (1800, 1200)

    sheet = pyvips.Image.new_from_file(str(tmp_path / "store/out/sheet.png"))
    assert (sheet.width, sheet.height, sheet.bands) == (1800, 1200, 4)
    assert round(sheet.xres * 25.4) == DPI
    # background stays transparent, the design is opaque
    assert sheet.getpoint(1790, 1190)[3] == 0
    assert sheet.getpoint(round(1.25 * DPI), round(1.75 * DPI))[3] == 255
    # rotated art: the red top band of the design ends up on the right edge (rot90 clockwise)
    top_right = sheet.getpoint(round(5.4 * DPI), round(1.25 * DPI))
    assert top_right[0] > 200 and top_right[2] < 100

    # QR codes in the label strips decode to the transfer ids
    pil = Image.open(tmp_path / "store/out/sheet.png").convert("RGB")
    found = {r.text for r in zxingcpp.read_barcodes(pil)}
    assert found == {p.transfer_id for p in placements}

    preview = pyvips.Image.new_from_file(str(tmp_path / "store/out/preview.png"))
    assert (preview.width, preview.height) == (300, 200)

    page = PdfReader(tmp_path / "store/out/sheet.pdf").pages[0]
    assert float(page.mediabox.width) / 72 == 6.0
    assert float(page.mediabox.height) / 72 == 4.0
    img = page["/Resources"]["/XObject"]["/Im0"]
    assert (img["/Width"], img["/Height"]) == (1800, 1200)


def test_long_sheet_pdf_uses_user_unit(tmp_path):
    from app.pdf import write_image_pdf

    src = tmp_path / "long.png"
    pyvips.Image.black(22, 300, bands=4).pngsave(str(src))
    write_image_pdf(src, tmp_path / "long.pdf", 22, 300)
    page = PdfReader(tmp_path / "long.pdf").pages[0]
    unit = float(page["/UserUnit"])
    assert float(page.mediabox.height) <= 14_400
    assert round(float(page.mediabox.height) * unit / 72, 3) == 300


def test_label_text():
    assert label_text(Label("A1", "2", "M", "", "Cactus")) == "A1 · 2 · M · Cactus"
