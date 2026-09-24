from pathlib import Path

import pytest
import pyvips
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.config import settings
from app.main import app
from tests.helpers import solid_design

ROOT = Path(settings.local_storage_dir)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
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

    r = client.post("/qa/clean-alpha", json={"file_key": "t/art.png", "out_key": "t/clean.png"})
    assert r.json() == {"key": "t/clean.png"}
    assert (
        client.post("/qa/check", json={"file_key": "t/clean.png"}).json()["soft_alpha_ratio"] == 0
    )

    r = client.post(
        "/nest",
        json={
            "items": [{"id": "d1", "width_in": 4, "height_in": 3, "quantity": 3}],
            "sheet_width_in": 10,
            "max_length_in": 30,
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
            "out_key": "t/sheet.png",
            "pdf_key": "t/sheet.pdf",
            "preview_key": "t/preview.png",
            "preview_width_px": 200,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width_px"] == 3000
    assert body["height_px"] == round(sheet["length_in"] * 300)
    assert body["bytes"] == (ROOT / "t/sheet.png").stat().st_size
    assert body["pdf_key"] == "t/sheet.pdf"
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
