"""B-80: PDF pages over 200in used to fall back to /UserUnit; the decision was to cap instead
(some RIPs/viewers don't honor /UserUnit, so a long sheet printed at the wrong size).
"""

import pytest
import pyvips
from pypdf import PdfReader

from app.pdf import PDF_MAX_LENGTH_IN, write_image_pdf
from app.vips import ImageError


def _src(tmp_path, w=22, h=100):
    path = tmp_path / "sheet.png"
    pyvips.Image.black(w, h, bands=4).pngsave(str(path))
    return path


def test_normal_sheet_pdf_page_size_matches_inches(tmp_path):
    src = _src(tmp_path, 22, 100)
    dest = tmp_path / "sheet.pdf"
    write_image_pdf(src, dest, 22, 100)
    page = PdfReader(dest).pages[0]
    assert "/UserUnit" not in page
    assert float(page.mediabox.width) / 72 == 22
    assert float(page.mediabox.height) / 72 == 100


def test_sheet_at_exactly_200in_is_allowed_without_user_unit(tmp_path):
    src = _src(tmp_path, 22, 1)
    dest = tmp_path / "sheet.pdf"
    write_image_pdf(src, dest, 22, PDF_MAX_LENGTH_IN)
    page = PdfReader(dest).pages[0]
    assert "/UserUnit" not in page
    assert float(page.mediabox.height) == PDF_MAX_LENGTH_IN * 72


def test_sheet_over_200in_is_rejected(tmp_path):
    src = _src(tmp_path, 22, 1)
    dest = tmp_path / "sheet.pdf"
    with pytest.raises(ImageError, match="200in"):
        write_image_pdf(src, dest, 22, 240)
    assert not dest.exists()


def test_wide_sheet_over_200in_is_also_rejected(tmp_path):
    # width_in, not just height_in, is checked -- a rotated/landscape page must be capped too.
    src = _src(tmp_path, 1, 22)
    dest = tmp_path / "sheet.pdf"
    with pytest.raises(ImageError, match="200in"):
        write_image_pdf(src, dest, 240, 22)
