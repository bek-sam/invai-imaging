"""PDF and SVG input at target DPI (T-9-1): dimensions and a visual checksum.

Rendering is deterministic (same pdfium/librsvg pins), so a fixed digest per fixture/DPI is a
valid assertion: it catches any change to how pixels decode, not just to width/height.
"""

import hashlib

import pytest
from reportlab.pdfgen import canvas

from app.pdf_input import rasterize_pdf
from app.vips import ImageError, load, to_srgba

SVG_WIDTH_IN, SVG_HEIGHT_IN = 2.0, 1.0
PDF_WIDTH_IN, PDF_HEIGHT_IN = 2.0, 1.0


def two_tone_svg(path, width_in: float = SVG_WIDTH_IN, height_in: float = SVG_HEIGHT_IN):
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width_in}in" height="{height_in}in"
     viewBox="0 0 {width_in} {height_in}">
  <rect x="0" y="0" width="{width_in / 2}" height="{height_in}" fill="#e02020"/>
  <rect x="{width_in / 2}" y="0" width="{width_in / 2}" height="{height_in}" fill="#20a040"/>
</svg>"""
    )
    return path


def two_tone_pdf(path, width_in: float = PDF_WIDTH_IN, height_in: float = PDF_HEIGHT_IN):
    from reportlab.lib.units import inch

    c = canvas.Canvas(str(path), pagesize=(width_in * inch, height_in * inch))
    c.setFillColorRGB(0.878, 0.125, 0.125)  # #e02020
    c.rect(0, 0, width_in / 2 * inch, height_in * inch, fill=1, stroke=0)
    c.setFillColorRGB(0.125, 0.627, 0.251)  # #20a040
    c.rect(width_in / 2 * inch, 0, width_in / 2 * inch, height_in * inch, fill=1, stroke=0)
    c.save()
    return path


def pixel_digest(img) -> str:
    return hashlib.sha256(to_srgba(img).write_to_memory()).hexdigest()


@pytest.mark.parametrize(
    "dpi,digest",
    [
        (150, "228b76fa1e5dbcc6fde86d05b08a2876b9d1f5ae3db8a2901d3a3ce6e539cfec"),
        (300, "14d01f39bd419dbf60eb4b56c582ea880952b97d18ccffdd75d509578d307c4c"),
    ],
)
def test_svg_rasterized_at_target_dpi(tmp_path, dpi, digest):
    svg = two_tone_svg(tmp_path / "design.svg")
    img = load(svg, target_dpi=dpi)
    assert (img.width, img.height) == (round(SVG_WIDTH_IN * dpi), round(SVG_HEIGHT_IN * dpi))
    assert pixel_digest(img) == digest


@pytest.mark.parametrize(
    "dpi,digest",
    [
        (150, "228b76fa1e5dbcc6fde86d05b08a2876b9d1f5ae3db8a2901d3a3ce6e539cfec"),
        (300, "14d01f39bd419dbf60eb4b56c582ea880952b97d18ccffdd75d509578d307c4c"),
    ],
)
def test_pdf_rasterized_at_target_dpi(tmp_path, dpi, digest):
    pdf = two_tone_pdf(tmp_path / "art.pdf")
    img = load(pdf, target_dpi=dpi)
    assert (img.width, img.height) == (round(PDF_WIDTH_IN * dpi), round(PDF_HEIGHT_IN * dpi))
    assert pixel_digest(img) == digest


def test_svg_without_target_dpi_is_unchanged(tmp_path):
    # Additive param: omitting it keeps every existing caller's (compose/render/qa) behaviour,
    # i.e. it must load exactly like `pyvips.Image.new_from_file` did before this card.
    import pyvips

    svg = two_tone_svg(tmp_path / "no_dpi_check.svg")
    via_load = load(svg)
    via_new_from_file = pyvips.Image.new_from_file(str(svg))
    assert (via_load.width, via_load.height) == (via_new_from_file.width, via_new_from_file.height)


def test_pdf_requires_target_dpi(tmp_path):
    pdf = two_tone_pdf(tmp_path / "art.pdf")
    with pytest.raises(ImageError, match="target_dpi"):
        load(pdf)


def test_pdf_multipage_is_rejected(tmp_path):
    from reportlab.lib.units import inch

    path = tmp_path / "multi.pdf"
    c = canvas.Canvas(str(path), pagesize=(2 * inch, 1 * inch))
    c.rect(0, 0, 2 * inch, 1 * inch, fill=1)
    c.showPage()
    c.rect(0, 0, 2 * inch, 1 * inch, fill=1)
    c.save()
    with pytest.raises(ImageError, match="single-page"):
        load(path, target_dpi=300)


def test_pdf_unreadable_raises_image_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nnot really a pdf")
    with pytest.raises(ImageError, match="unreadable pdf"):
        rasterize_pdf(bad, 300)
