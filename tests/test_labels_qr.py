from io import BytesIO

import zxingcpp
from pypdf import PdfReader

from app.labels import QrLabelItem, render_qr_labels


def _decode_page_qr(page) -> str:
    images = list(page.images)
    assert images, "no embedded QR image on page"
    im = images[0].image.convert("RGB")
    results = zxingcpp.read_barcodes(im)
    assert results, "QR did not decode"
    return results[0].text


def test_renders_one_page_per_item_at_the_right_size():
    pdf = render_qr_labels(
        [
            QrLabelItem(code="BIN:A1", caption="Bin A1", size="4x6"),
            QrLabelItem(
                code="B:00000000-0000-0000-0000-000000000001", caption="Blank M Black", size="2x1"
            ),
        ]
    )
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) == 2

    p0 = reader.pages[0]
    assert (round(float(p0.mediabox.width) / 72, 2), round(float(p0.mediabox.height) / 72, 2)) == (
        4.0,
        6.0,
    )
    assert _decode_page_qr(p0) == "BIN:A1"

    p1 = reader.pages[1]
    assert (round(float(p1.mediabox.width) / 72, 2), round(float(p1.mediabox.height) / 72, 2)) == (
        2.0,
        1.0,
    )
    assert _decode_page_qr(p1) == "B:00000000-0000-0000-0000-000000000001"


def test_scans_without_a_caption():
    pdf = render_qr_labels([QrLabelItem(code="BIN:NOCAP", caption=None, size="2x1")])
    reader = PdfReader(BytesIO(pdf))
    assert _decode_page_qr(reader.pages[0]) == "BIN:NOCAP"


def test_rejects_empty_input():
    try:
        render_qr_labels([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
