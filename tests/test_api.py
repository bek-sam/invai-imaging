from pathlib import Path

import pytest
import pyvips
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from app.config import settings
from app.main import app
from tests.helpers import solid_design

ROOT = Path(settings.local_storage_dir)


@pytest.fixture(scope="module")
def client():
    # Every call but /health needs the shared secret (T-9-5); the dev default is used here.
    with TestClient(app, headers={"X-Imaging-Secret": settings.shared_secrets()[0]}) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["vips_version"].startswith("8.")


def test_full_flow(client):
    r = client.post(
        "/sample-art",
        json={
            "text": "Desert Bloom",
            "out_key": "t/art.png",
            "width_in": 4,
            "height_in": 3,
            "color_hex": "#E4572E",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"key": "t/art.png"}
    art = pyvips.Image.new_from_file(str(ROOT / "t/art.png"))
    assert (art.width, art.height, art.bands) == (1200, 900, 4)

    r = client.post("/qa/check", json={"file_key": "t/art.png", "target_width_in": 4})
    qa = r.json()
    assert qa["effective_dpi"] == 300
    assert qa["has_alpha"] is True
    assert not [i for i in qa["issues"] if i["severity"] == "error"]

    # A PDF design upload goes through /qa/check the same way a PNG does (gate finding B: this
    # used to fail 100% of the time with "pdf input requires target_dpi").
    (ROOT / "t").mkdir(parents=True, exist_ok=True)
    pdf_path = ROOT / "t/design.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(2 * inch, 1 * inch))
    c.setFillColorRGB(0.878, 0.125, 0.125)
    margin = 0.1 * inch
    c.rect(margin, margin, 2 * inch - 2 * margin, 1 * inch - 2 * margin, fill=1)
    c.save()
    r = client.post(
        "/qa/check",
        json={"file_key": "t/design.pdf", "target_width_in": 2, "target_height_in": 1},
    )
    assert r.status_code == 200, r.text
    pdf_qa = r.json()
    assert not [i for i in pdf_qa["issues"] if i["severity"] == "error"]
    assert pdf_qa["effective_dpi"] == 300  # QA's own default DPI for vector art (DPI_WARN)

    # An explicit target_dpi rasterizes the PDF at that DPI instead.
    r = client.post(
        "/qa/check",
        json={
            "file_key": "t/design.pdf",
            "target_width_in": 2,
            "target_height_in": 1,
            "target_dpi": 150,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["width_px"] == 300

    r = client.post("/qa/clean-alpha", json={"file_key": "t/art.png", "out_key": "t/clean.png"})
    assert r.json() == {"key": "t/clean.png"}
    assert (
        client.post("/qa/check", json={"file_key": "t/clean.png"}).json()["soft_alpha_ratio"] == 0
    )

    # label_height_in must match between /nest and /compose (README); 0.8in clears the AC1
    # 0.5mm/module QR floor at compose's default label_gap_in (0.125in) -- see test_compose.py.
    label_height_in = 0.8
    r = client.post(
        "/nest",
        json={
            "items": [{"id": "d1", "width_in": 4, "height_in": 3, "quantity": 3}],
            "sheet_width_in": 10,
            "max_length_in": 30,
            "label_height_in": label_height_in,
        },
    )
    sheets = r.json()["sheets"]
    assert len(sheets) == 1 and len(sheets[0]["placements"]) == 3
    sheet = sheets[0]

    r = client.post(
        "/compose",
        json={
            "width_in": 10,
            "length_in": sheet["length_in"],
            "placements": [
                {
                    "transfer_id": f"tr-{p['copy']}",
                    "file_key": "t/art.png",
                    **{k: p[k] for k in ("x_in", "y_in", "width_in", "height_in", "rotated")},
                    "label": {
                        "order_no": "DB-1",
                        "item_no": str(p["copy"] + 1),
                        "size": "M",
                        "color": "Black",
                        "design": "Desert Bloom",
                        "reprint": False,
                    },
                }
                for p in sheet["placements"]
            ],
            "label_height_in": label_height_in,
            "out_key": "t/sheet.png",
            "pdf_key": "t/sheet.pdf",
            "preview_key": "t/preview.png",
            "preview_width_px": 200,
            "filename_hint": "SH-9001_DB-1",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width_px"] == 3000
    assert body["height_px"] == round(sheet["length_in"] * 300)
    assert body["bytes"] == (ROOT / "t/sheet.png").stat().st_size
    assert body["pdf_key"] == "t/sheet.pdf"
    # AC2's ContentDisposition (download filename) is S3-only; covered at the storage layer in
    # tests/test_storage.py::test_s3_storage_sets_content_disposition_from_filename_hint. The
    # header QR itself is covered directly (with enough top clearance to actually fit) in
    # tests/test_compose.py::test_compose_header_decodes; here the sheet's real 0.25in margin
    # isn't tall enough for one, so `filename_hint` is exercised end-to-end without asserting a
    # header decode -- compose must still succeed (it's best-effort, never a failure).
    assert pyvips.Image.new_from_file(str(ROOT / "t/preview.png")).width == 200

    r = client.post(
        "/mockup",
        json={
            "design_key": "t/art.png",
            "blank_color_hex": "#1E3A5F",
            "placement": "front",
            "out_key": "t/mock.png",
            "size_px": 400,
        },
    )
    assert r.json() == {"key": "t/mock.png"}
    assert pyvips.Image.new_from_file(str(ROOT / "t/mock.png")).width == 400

    r = client.post(
        "/labels/mock",
        json={
            "shipment_id": "sh_1",
            "carrier": "InvAI Mock Post",
            "service": "Ground Advantage",
            "tracking_code": "9400100000000000000001",
            "from": {"name": "Desert Bloom Tees", "city": "Tucson", "state": "AZ", "zip": "85701"},
            "to": {"name": "Jane Doe", "city": "Austin", "state": "TX", "zip": "78701"},
            "weight_oz": 6.2,
            "out_key": "t/label.pdf",
        },
    )
    assert r.json() == {"key": "t/label.pdf"}
    page = PdfReader(ROOT / "t/label.pdf").pages[0]
    assert (float(page.mediabox.width), float(page.mediabox.height)) == (288, 432)
    assert "NOT A REAL LABEL" in page.extract_text()

    r = client.post(
        "/labels/qr",
        json={
            "labels": [
                {"code": "BIN:A1", "caption": "Bin A1", "size": "4x6"},
                {"code": "B:00000000-0000-0000-0000-000000000001", "size": "2x1"},
            ],
            "out_key": "t/qr-labels.pdf",
        },
    )
    assert r.json() == {"key": "t/qr-labels.pdf"}
    pages = PdfReader(ROOT / "t/qr-labels.pdf").pages
    assert len(pages) == 2
    assert (float(pages[0].mediabox.width), float(pages[0].mediabox.height)) == (288, 432)
    assert (float(pages[1].mediabox.width), float(pages[1].mediabox.height)) == (144, 72)


def test_preview_endpoint(client):
    (ROOT / "t").mkdir(parents=True, exist_ok=True)
    # Downscaled: longest side capped at max_px, aspect kept, sRGB with alpha.
    solid_design(ROOT / "t/preview-src.png", 800, 400)
    r = client.post(
        "/preview", json={"file_key": "t/preview-src.png", "out_key": "t/preview-out.png"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"out_key": "t/preview-out.png", "width_px": 512, "height_px": 256}
    img = pyvips.Image.new_from_file(str(ROOT / "t/preview-out.png"))
    assert (img.width, img.height, img.bands) == (512, 256, 4)
    assert img.interpretation == "srgb"

    # Never upscaled: a source already smaller than max_px keeps its native size.
    solid_design(ROOT / "t/preview-small.png", 100, 50)
    r = client.post(
        "/preview",
        json={
            "file_key": "t/preview-small.png",
            "out_key": "t/preview-small-out.png",
            "max_px": 512,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"out_key": "t/preview-small-out.png", "width_px": 100, "height_px": 50}

    # A PDF's first page previews the same way (gate convention: vector art rasterizes at
    # qa.DPI_WARN, same as /qa/check).
    pdf_path = ROOT / "t/preview.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(2 * inch, 1 * inch))
    c.setFillColorRGB(0.2, 0.4, 0.8)
    c.rect(0, 0, 2 * inch, 1 * inch, fill=1)
    c.save()
    r = client.post(
        "/preview", json={"file_key": "t/preview.pdf", "out_key": "t/preview-pdf-out.png"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width_px"] / body["height_px"] == pytest.approx(2.0, rel=0.01)

    r = client.post("/preview", json={"file_key": "t/missing.png", "out_key": "t/x.png"})
    assert r.status_code == 422
    assert "missing.png" in r.json()["detail"]

    r = client.post(
        "/preview", json={"file_key": "t/preview-src.png", "out_key": "t/x.png", "max_px": 10}
    )
    assert r.status_code == 422


def test_personalization_endpoint(client):
    r = client.post(
        "/render/personalization",
        json={
            "template": {
                "width_in": 6,
                "height_in": 3,
                "slots": [
                    {
                        "name": "name",
                        "kind": "text",
                        "x_in": 0,
                        "y_in": 0,
                        "w_in": 6,
                        "h_in": 1.5,
                        "font_family": "Pacifico",
                        "font_size_pt": 60,
                        "color": "#222222",
                        "align": "center",
                        "uppercase": False,
                    },
                    {
                        "name": "year",
                        "kind": "text",
                        "x_in": 0,
                        "y_in": 1.5,
                        "w_in": 6,
                        "h_in": 1.5,
                        "font_family": "Bebas Neue",
                        "font_size_pt": 48,
                        "color": "#222222",
                        "align": "center",
                    },
                ],
            },
            "values": {"name": "Olivia"},
            "out_key": "t/pers.png",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["width_px"], body["height_px"]) == (1800, 900)
    assert body["flags"] == [{"slot": "year", "code": "empty", "message": "no value given"}]


def test_personalization_rejects_unknown_font(client):
    r = client.post(
        "/render/personalization",
        json={
            "template": {
                "width_in": 4,
                "height_in": 2,
                "slots": [
                    {
                        "name": "name",
                        "x_in": 0,
                        "y_in": 0,
                        "w_in": 4,
                        "h_in": 2,
                        "font_family": "Comic Sans",
                    }
                ],
            },
            "values": {"name": "Olivia"},
            "out_key": "t/bad-font.png",
        },
    )
    assert r.status_code == 422
    assert "unknown font family" in r.json()["detail"]


def test_personalization_rejects_duplicate_slot_names(client):
    slot = {"name": "name", "x_in": 0, "y_in": 0, "w_in": 2, "h_in": 1}
    r = client.post(
        "/render/personalization",
        json={
            "template": {"width_in": 4, "height_in": 2, "slots": [slot, dict(slot)]},
            "values": {},
            "out_key": "t/dup.png",
        },
    )
    assert r.status_code == 422
    assert "duplicate slot name" in r.json()["detail"]


def test_personalization_rejects_slot_outside_template_bounds(client):
    slot = {"name": "name", "x_in": 3, "y_in": 0, "w_in": 3, "h_in": 1}
    r = client.post(
        "/render/personalization",
        json={
            "template": {"width_in": 4, "height_in": 2, "slots": [slot]},
            "values": {},
            "out_key": "t/outside.png",
        },
    )
    assert r.status_code == 422
    assert "extends outside" in r.json()["detail"]


def test_personalization_photo_slot_end_to_end(client):
    photo_path = ROOT / "t/buyer-photo.jpg"
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (1200, 1200), (10, 20, 30)).save(photo_path)

    r = client.post(
        "/render/personalization",
        json={
            "template": {
                "width_in": 3,
                "height_in": 3,
                "slots": [
                    {
                        "name": "photo",
                        "kind": "photo",
                        "x_in": 0,
                        "y_in": 0,
                        "w_in": 3,
                        "h_in": 3,
                        "fit": "fill",
                    }
                ],
            },
            "values": {"photo": "t/buyer-photo.jpg"},
            "out_key": "t/photo-slot.png",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["flags"] == []
    img = pyvips.Image.new_from_file(str(ROOT / "t/photo-slot.png"))
    assert (img.width, img.height) == (900, 900)
    assert img.crop(0, 0, 10, 10)[3].min() == 255  # filled, fully opaque


def test_errors_are_422_with_detail(client):
    r = client.post("/qa/check", json={"file_key": "nope/missing.png"})
    assert r.status_code == 422
    assert "missing.png" in r.json()["detail"]

    r = client.post("/nest", json={"items": [{"id": "x", "width_in": 40, "height_in": 40}]})
    assert r.status_code == 422
    assert "does not fit" in r.json()["detail"]

    r = client.post("/nest", json={"items": [{"id": "x", "width_in": -1, "height_in": 4}]})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)

    solid_design(ROOT / "t_err.png", 100, 100)
    r = client.post(
        "/compose",
        json={
            "width_in": 2,
            "length_in": 2,
            "out_key": "x.png",
            "preview_key": "xp.png",
            "placements": [
                {
                    "transfer_id": "t",
                    "file_key": "t_err.png",
                    "x_in": 1.5,
                    "y_in": 0,
                    "width_in": 1,
                    "height_in": 1,
                }
            ],
        },
    )
    assert r.status_code == 422
    assert "outside the sheet" in r.json()["detail"]


def test_sample_art_every_style_fits_narrow_sleeve_art(client):
    # Each text hashes to a different style; a 3x10 in sleeve must not break any of them.
    for i, text in enumerate(["Tucson Old Pueblo", "Lake Powell Weekend", "Howdy", "Rodeo Nights"]):
        r = client.post(
            "/sample-art",
            json={
                "text": text,
                "out_key": f"t/sleeve{i}.png",
                "width_in": 3,
                "height_in": 10,
                "dpi": 100,
                "color_hex": "#9c5b3a",
            },
        )
        assert r.status_code == 200, (text, r.text)


def test_compose_rejects_pdf_output_over_200in(client):
    # B-80: PDF pages can't rely on /UserUnit, so /compose caps PDF output at 200in with a clear
    # 422 -- checked here before any compositing work, not just inside app/pdf.py.
    r = client.post(
        "/compose",
        json={
            "width_in": 22,
            "length_in": 240,
            "placements": [],
            "out_key": "t/long.png",
            "pdf_key": "t/long.pdf",
            "preview_key": "t/long-preview.png",
        },
    )
    assert r.status_code == 422
    assert "200in" in r.json()["detail"]
    assert not (ROOT / "t/long.pdf").exists()

    # PNG-only output at the same length is unaffected by the PDF cap (kept tiny: width/dpi don't
    # matter to this check, only length_in and pdf_key do).
    r = client.post(
        "/compose",
        json={
            "width_in": 1,
            "length_in": 240,
            "dpi": 36,
            "placements": [],
            "out_key": "t/long-ok.png",
            "preview_key": "t/long-ok-preview.png",
        },
    )
    assert r.status_code == 200, r.text
