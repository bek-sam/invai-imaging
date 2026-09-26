"""T-9-5 (B-19): service auth, heavy-job concurrency limit, input bounds, format allowlist,
decode pixel cap (incl. the PDF page budget), dev-only endpoints and render timing."""

import struct
import time
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import pdf_input
from app.config import DEV_SHARED_SECRET, Settings, production_config_error, settings
from app.guard import limiter, secret_ok
from app.limits import declared_type_error, sniff_format
from app.main import MAX_PLACEMENTS, app
from app.vips import ImageError, load
from tests.helpers import solid_design

ROOT = Path(settings.local_storage_dir)
AUTH = {"X-Imaging-Secret": DEV_SHARED_SECRET}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers=AUTH) as c:
        yield c


@pytest.fixture(scope="module")
def anon():
    with TestClient(app) as c:
        yield c


def _png_header_only(path: Path, w: int, h: int) -> Path:
    """A PNG that claims w x h in its header with (almost) no pixel data: a decompression bomb
    stand-in that costs nothing to make."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    body = zlib.compress(b"\x00" * 16)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", body) + chunk(b"IEND", b"")
    )
    return path


def _compose_body(**over) -> dict:
    body = {
        "width_in": 22,
        "length_in": 10,
        "dpi": 72,
        "placements": [],
        "out_key": "t/limits/sheet.png",
        "preview_key": "t/limits/sheet-preview.png",
    }
    body.update(over)
    return body


# --- service auth (AC3) --------------------------------------------------------------------


def test_missing_auth_header_is_401(anon):
    r = anon.post("/nest", json={"items": [{"id": "a", "width_in": 2, "height_in": 2}]})
    assert r.status_code == 401
    assert r.json() == {"detail": "missing or invalid imaging credentials"}


def test_wrong_secret_is_401(anon):
    r = anon.post(
        "/nest",
        json={"items": [{"id": "a", "width_in": 2, "height_in": 2}]},
        headers={"X-Imaging-Secret": DEV_SHARED_SECRET + "x"},
    )
    assert r.status_code == 401


def test_right_secret_passes_and_health_is_open(client, anon):
    r = client.post("/nest", json={"items": [{"id": "a", "width_in": 2, "height_in": 2}]})
    assert r.status_code == 200, r.text
    assert int(r.headers["x-render-ms"]) >= 0  # AC5: render time on every response
    assert anon.get("/health").status_code == 200


def test_rejected_before_body_is_parsed(anon):
    # Invalid JSON would be a 422 if the body were read; auth answers first.
    r = anon.post("/compose", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 401


def test_secret_rotation_list(monkeypatch):
    monkeypatch.setattr(settings, "imaging_shared_secret", "a" * 32 + ", " + "b" * 32)
    assert secret_ok("a" * 32) and secret_ok("b" * 32)
    assert not secret_ok("c" * 32) and not secret_ok("") and not secret_ok(None)


def test_production_requires_a_real_secret():
    dev = Settings(imaging_env="development")
    assert production_config_error(dev) is None
    assert "real secret" in production_config_error(Settings(imaging_env="production"))
    short = Settings(imaging_env="production", imaging_shared_secret="short")
    assert "32 characters" in production_config_error(short)
    real = Settings(imaging_env="production", imaging_shared_secret="s" * 40)
    assert production_config_error(real) is None


def test_production_refuses_to_start_on_dev_secret(monkeypatch):
    monkeypatch.setattr(settings, "imaging_env", "production")
    with pytest.raises(RuntimeError, match="IMAGING_SHARED_SECRET"), TestClient(app):
        pass


# --- concurrency limit (AC1) ---------------------------------------------------------------


def test_heavy_jobs_over_the_limit_get_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "imaging_max_heavy_jobs", 1)
    monkeypatch.setattr(settings, "imaging_retry_after_s", 7)
    assert limiter.try_acquire()  # one compose already running
    try:
        r = client.post("/compose", json=_compose_body())
        assert r.status_code == 429
        assert r.headers["retry-after"] == "7"
        r = client.post("/render/personalization", json={})
        assert r.status_code == 429
        # Light endpoints aren't limited.
        r = client.post("/nest", json={"items": [{"id": "a", "width_in": 2, "height_in": 2}]})
        assert r.status_code == 200
    finally:
        limiter.release()
    r = client.post("/compose", json=_compose_body())
    assert r.status_code == 200, r.text
    assert r.json()["render_ms"] >= 0
    assert limiter.in_flight == 0  # the slot is released after the response


def test_slot_released_when_the_job_fails(client, monkeypatch):
    monkeypatch.setattr(settings, "imaging_max_heavy_jobs", 1)
    r = client.post("/compose", json=_compose_body(width_in=-1))
    assert r.status_code == 422
    assert limiter.in_flight == 0


# --- input bounds (AC2) --------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"width_in": 60.5},
        {"length_in": 241},
        {"dpi": 1201},
        {"dpi": 35},
        {"out_key": "k" * 513},
        {
            "placements": [
                {
                    "transfer_id": "t",
                    "file_key": "k.png",
                    "x_in": -1,
                    "y_in": 0,
                    "width_in": 1,
                    "height_in": 1,
                }
            ]
        },
    ],
)
def test_compose_bounds(client, over):
    r = client.post("/compose", json=_compose_body(**over))
    assert r.status_code == 422, r.text


def test_compose_placement_count_is_capped(client):
    p = {"transfer_id": "t", "file_key": "k.png", "x_in": 0, "y_in": 0, "width_in": 1}
    p["height_in"] = 1
    r = client.post("/compose", json=_compose_body(placements=[p] * (MAX_PLACEMENTS + 1)))
    assert r.status_code == 422
    assert "placements" in r.json()["detail"]


def test_nest_bounds(client):
    item = {"id": "a", "width_in": 2, "height_in": 2}
    assert client.post("/nest", json={"items": [item], "sheet_width_in": 61}).status_code == 422
    many = {**item, "quantity": 10_000}
    r = client.post("/nest", json={"items": [many, many]})
    assert r.status_code == 422
    assert "copies" in r.json()["detail"]


def test_personalization_bounds(client):
    slot = {"name": "n", "x_in": 0, "y_in": 0, "w_in": 1, "h_in": 1}
    body = {
        "template": {"width_in": 4, "height_in": 4, "slots": [slot]},
        "values": {"n": "x" * 1001},
        "out_key": "t/limits/p.png",
    }
    assert client.post("/render/personalization", json=body).status_code == 422
    body["values"] = {"n": "Ana"}
    body["dpi"] = 1201
    assert client.post("/render/personalization", json=body).status_code == 422
    body["dpi"] = 72
    r = client.post("/render/personalization", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["render_ms"] >= 0


# --- decode pixel cap and format allowlist (AC2, AC7) --------------------------------------


def test_oversized_png_is_refused_from_its_header(tmp_path):
    bomb = _png_header_only(tmp_path / "bomb", 100_000, 100_000)  # 10 G pixels
    started = time.perf_counter()
    with pytest.raises(ImageError, match="pixel limit"):
        load(bomb)
    assert time.perf_counter() - started < 1  # refused from the header, nothing decoded


def test_pixel_cap_comes_from_config(tmp_path, monkeypatch):
    art = solid_design(tmp_path / "art.png", 40, 30)
    assert load(art).width == 40
    monkeypatch.setattr(settings, "imaging_max_decode_pixels", 1_000)
    with pytest.raises(ImageError, match="40 x 30 px"):
        load(art)


def test_oversized_svg_is_refused_before_render(tmp_path):
    svg = tmp_path / "big.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000in" height="1000in"/>')
    with pytest.raises(ImageError, match="pixel limit"):
        load(svg, target_dpi=300)


def test_pdf_page_over_budget_is_refused_before_render(tmp_path, monkeypatch):
    path = tmp_path / "huge.pdf"
    c = canvas.Canvas(str(path), pagesize=(200 * 72, 200 * 72))  # 200 x 200 in page
    c.drawString(10, 10, "big")
    c.save()
    rendered = []
    monkeypatch.setattr(
        pdf_input.pdfium.PdfPage, "render", lambda *a, **k: rendered.append(1) or 1 / 0
    )
    with pytest.raises(ImageError, match=r"pdf page \(200.00 x 200.00 in at 300 DPI\)"):
        load(path, target_dpi=300)
    assert rendered == []  # never rasterized
    monkeypatch.undo()
    # The same page at a DPI that fits the budget still renders.
    monkeypatch.setattr(settings, "imaging_max_decode_pixels", 10_000_000)
    assert load(path, target_dpi=10).width == 2000


@pytest.mark.parametrize(
    "head",
    [b"GIF89a" + b"\x00" * 32, b"BM" + b"\x00" * 32, b"\x00\x00\x00\x0cjP  " + b"\x00" * 32],
)
def test_formats_outside_the_allowlist_are_refused(tmp_path, head):
    path = tmp_path / "x"
    path.write_bytes(head)
    assert sniff_format(path) is None
    with pytest.raises(ImageError, match="unsupported image format"):
        load(path)


def test_magic_bytes_must_match_the_declared_type(tmp_path):
    png = solid_design(tmp_path / "art.png", 20, 20)
    assert declared_type_error("co/art/2026/09/x.png", png) is None
    assert declared_type_error("co/art/2026/09/x", png) is None  # no declared type
    assert "declares JPEG" in declared_type_error("co/art/2026/09/x.jpg", png)
    assert "unsupported" in declared_type_error("co/art/2026/09/x.gif", png)


def test_wrong_format_through_the_api(client):
    (ROOT / "t/limits").mkdir(parents=True, exist_ok=True)
    solid_design(ROOT / "t/limits/src.png", 40, 40).rename(ROOT / "t/limits/renamed.jpg")
    (ROOT / "t/limits/anim.gif").write_bytes(b"GIF89a" + b"\x00" * 64)

    # QA reports it like any unreadable file (200 + `unreadable` issue).
    r = client.post("/qa/check", json={"file_key": "t/limits/renamed.jpg"})
    assert r.status_code == 200
    assert r.json()["issues"][0]["code"] == "unreadable"
    assert "declares JPEG" in r.json()["issues"][0]["message"]

    # Everything else is a 422, including compose (which downloads through its own storage).
    r = client.post(
        "/mockup",
        json={"design_key": "t/limits/anim.gif", "blank_color_hex": "#fff", "out_key": "t/m.png"},
    )
    assert r.status_code == 422
    assert "unsupported image format" in r.json()["detail"]
    placement = {
        "transfer_id": "t1",
        "file_key": "t/limits/renamed.jpg",
        "x_in": 1,
        "y_in": 1,
        "width_in": 2,
        "height_in": 2,
    }
    r = client.post("/compose", json=_compose_body(placements=[placement]))
    assert r.status_code == 422
    assert "declares JPEG" in r.json()["detail"]


# --- dev-only endpoints (AC4) --------------------------------------------------------------


def test_dev_endpoints_are_404_unless_the_flag_is_set(client, monkeypatch):
    monkeypatch.setattr(settings, "imaging_dev_endpoints", False)
    art = {"text": "x", "out_key": "t/limits/a.png", "width_in": 1, "height_in": 1}
    assert client.post("/sample-art", json=art).status_code == 404
    assert client.post("/labels/mock", json={}).status_code == 404
    monkeypatch.setattr(settings, "imaging_dev_endpoints", True)
    assert client.post("/sample-art", json={**art, "dpi": 72}).status_code == 200
