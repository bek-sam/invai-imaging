"""invai-imaging HTTP API (v1-plan.md section 5.1). Inputs and outputs are storage keys."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pyvips
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app import compose, fonts, labels, mockups, nesting, qa, render, sample_art
from app.config import production_config_error, settings
from app.guard import GuardMiddleware, limiter
from app.limits import declared_type_error
from app.pdf import PDF_MAX_LENGTH_IN
from app.storage import NotFoundError, S3Storage, Storage, get_storage
from app.vips import ImageError, load, save_png, to_srgba, workdir

log = logging.getLogger("invai.imaging")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if err := production_config_error(settings):
        raise RuntimeError(f"Refusing to start: {err}")
    storage = get_storage()
    if isinstance(storage, S3Storage) and settings.s3_create_bucket:
        try:
            storage.ensure_bucket()
        except (BotoCoreError, ClientError) as e:  # keep serving; requests report the error
            log.warning("could not ensure bucket %s: %s", settings.s3_bucket, e)
    yield
    # uvicorn has stopped accepting and waited (--timeout-graceful-shutdown) for requests.
    if limiter.in_flight:
        log.warning("shutting down with %d heavy jobs still running", limiter.in_flight)


app = FastAPI(title="invai-imaging", lifespan=lifespan)
# Service auth + heavy-job limit + X-Render-Ms, before any body is read (app/guard.py).
app.add_middleware(GuardMiddleware)

# Storage keys: same shape as the backend's isSafeKey (512 chars max).
Key = Field(min_length=1, max_length=512)

# Most placements one compose (or copies one nest) may carry: a 22 x 240in sheet of 1in
# designs is about 2,500, and 4,000 pieces nest in 0.3 s (audit A-INF).
MAX_PLACEMENTS = 10_000


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts)})


@app.exception_handler(ImageError)
async def image_error(_: Request, exc: ImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(NotFoundError)
async def not_found(_: Request, exc: NotFoundError):
    return JSONResponse(status_code=422, content={"detail": f"file not found: {exc}"})


@app.exception_handler(nesting.NestError)
async def nest_error(_: Request, exc: nesting.NestError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


class _CheckedStorage:
    """Storage whose downloads must match the type their key declares (`<uuid>.png` must be a
    PNG by magic bytes), so a renamed file is refused before any decoder sees it."""

    def __init__(self, inner: Storage) -> None:
        self._inner = inner

    def download(self, key: str, dest: Path) -> None:
        self._inner.download(key, dest)
        if err := declared_type_error(key, dest):
            raise ImageError(err)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _download(key: str, dest: Path) -> Path:
    _CheckedStorage(get_storage()).download(key, dest)
    return dest


def _dev_only() -> None:
    """Mock/test endpoints answer 404 unless IMAGING_DEV_ENDPOINTS is set."""
    if not settings.imaging_dev_endpoints:
        raise HTTPException(status_code=404, detail="Not Found")


# --- health -------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True, "vips_version": ".".join(str(pyvips.version(i)) for i in range(3))}


# --- QA -----------------------------------------------------------------------------------


class QACheckRequest(BaseModel):
    file_key: str = Key
    target_width_in: float | None = Field(None, gt=0, le=240)
    target_height_in: float | None = Field(None, gt=0, le=240)
    # PDF/SVG art has no native pixel size; this is the DPI QA rasterizes it at before
    # measuring. Optional: omit it to get the shop's own recommended DPI (T-9-1 gate fix).
    target_dpi: float | None = Field(None, ge=36, le=1200)


@app.post("/qa/check")
def qa_check(req: QACheckRequest) -> dict:
    with workdir() as tmp:
        path = tmp / "input"
        get_storage().download(req.file_key, path)
        # A file that doesn't match its key's type is reported like any unreadable file.
        if err := declared_type_error(req.file_key, path):
            return qa.QAResult(issues=[qa.Issue("unreadable", "error", err)]).to_dict()
        return qa.check(path, req.target_width_in, req.target_height_in, req.target_dpi).to_dict()


class CleanAlphaRequest(BaseModel):
    file_key: str = Key
    out_key: str = Key
    threshold: int = Field(128, ge=1, le=255)


@app.post("/qa/clean-alpha")
def qa_clean_alpha(req: CleanAlphaRequest) -> dict:
    with workdir() as tmp:
        src = _download(req.file_key, tmp / "input")
        out = tmp / "clean.png"
        qa.clean_alpha(src, out, req.threshold)
        get_storage().upload(out, req.out_key)
    return {"key": req.out_key}


# --- personalization ----------------------------------------------------------------------


class SlotModel(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    kind: Literal["text", "photo"] = "text"
    x_in: float = Field(ge=0)
    y_in: float = Field(ge=0)
    w_in: float = Field(gt=0)
    h_in: float = Field(gt=0)
    font_family: str = "Inter"
    font_size_pt: float = Field(36, gt=0, le=1000)
    min_font_size_pt: float | None = Field(None, gt=0, le=1000)
    max_lines: int | None = Field(None, gt=0, le=50)
    stroke_width_pt: float = Field(0, ge=0, le=100)
    stroke_color: str | None = None
    fit: Literal["fit", "fill"] = "fit"
    color: str = "#000000"
    align: Literal["left", "center", "right"] = "center"
    max_chars: int | None = Field(None, gt=0)
    uppercase: bool = False

    @field_validator("font_family")
    @classmethod
    def _known_font(cls, v: str) -> str:
        # The real B-81 bug: fonts.normalize_family() used to silently fall back to Inter on an
        # unknown name instead of rejecting it. Reject here, before it ever reaches render.py.
        if not fonts.known_family(v):
            raise ValueError(f"unknown font family: {v!r}")
        return v


class TemplateModel(BaseModel):
    width_in: float = Field(gt=0, le=60)
    height_in: float = Field(gt=0, le=60)
    background_key: str | None = Field(None, min_length=1, max_length=512)
    slots: list[SlotModel] = Field(max_length=50)

    @model_validator(mode="after")
    def _check_geometry(self) -> "TemplateModel":
        names: set[str] = set()
        for s in self.slots:
            if s.name in names:
                raise ValueError(f"duplicate slot name: {s.name!r}")
            names.add(s.name)
            if s.x_in + s.w_in > self.width_in + 1e-6 or s.y_in + s.h_in > self.height_in + 1e-6:
                raise ValueError(
                    f"slot {s.name!r} extends outside the {self.width_in}x{self.height_in} "
                    "in template"
                )
        return self


class PersonalizationRequest(BaseModel):
    template: TemplateModel
    values: dict[str, str] = Field(default_factory=dict, max_length=50)
    out_key: str = Key
    dpi: int = Field(300, ge=36, le=1200)

    @field_validator("values")
    @classmethod
    def _bounded_values(cls, v: dict[str, str]) -> dict[str, str]:
        for name, value in v.items():
            if len(name) > 40 or len(value) > 1000:
                raise ValueError(f"value for {name[:40]!r} is too long (max 1000 characters)")
        return v


@app.post("/render/personalization")
def render_personalization(req: PersonalizationRequest) -> dict:
    t = req.template
    slots = [render.Slot(**s.model_dump()) for s in t.slots]
    started = time.perf_counter()
    with workdir() as tmp:
        bg = _download(t.background_key, tmp / "background") if t.background_key else None
        # Photo slots carry a storage key in `values` (like `background_key`), not literal text.
        photos = {
            s.name: _download(req.values[s.name], tmp / f"photo_{s.name}")
            for s in t.slots
            if s.kind == "photo" and req.values.get(s.name)
        }
        out = tmp / "out.png"
        res = render.render_personalization(
            t.width_in, t.height_in, slots, req.values, req.dpi, out, bg, photos
        )
        get_storage().upload(out, req.out_key)
    return {
        "key": req.out_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "flags": [f.__dict__ for f in res.flags],
        "render_ms": round((time.perf_counter() - started) * 1000),
    }


# --- preview --------------------------------------------------------------------------------


class PreviewRequest(BaseModel):
    file_key: str = Key
    out_key: str = Key
    max_px: int = Field(512, ge=64, le=2048)


@app.post("/preview")
def preview(req: PreviewRequest) -> dict:
    """A small sRGB PNG thumbnail of any supported input (design art, a PDF's first page, ...),
    for catalog/order-item previews (T-P1-4). Longest side is at most `max_px`, never upscaled."""
    with workdir() as tmp:
        src = _download(req.file_key, tmp / "input")
        # Vector input (PDF/SVG) has no native pixel size; rasterize at QA's own recommended
        # print DPI (qa.DPI_WARN), same convention as /qa/check. Raster formats ignore target_dpi.
        img = to_srgba(load(src, access="sequential", target_dpi=qa.DPI_WARN))
        scale = min(1.0, req.max_px / max(img.width, img.height))
        if scale < 1.0:
            img = img.premultiply().resize(scale, kernel="lanczos3").unpremultiply().cast("uchar")
        out = tmp / "preview.png"
        save_png(img, out)
        get_storage().upload(out, req.out_key)
    return {"out_key": req.out_key, "width_px": img.width, "height_px": img.height}


# --- nesting ------------------------------------------------------------------------------


class NestItemModel(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    width_in: float = Field(gt=0, le=240)
    height_in: float = Field(gt=0, le=240)
    quantity: int = Field(1, ge=1, le=10_000)


class NestRequest(BaseModel):
    items: list[NestItemModel] = Field(max_length=5_000)
    sheet_width_in: float = Field(22.0, gt=0, le=60)
    spacing_in: float = Field(0.25, ge=0, le=6)
    margin_in: float = Field(0.25, ge=0, le=6)
    max_length_in: float = Field(240.0, gt=0, le=240)
    allow_rotation: bool = True
    label_height_in: float = Field(0.42, ge=0, le=6)
    header_height_in: float = Field(0.45, ge=0, le=6)

    @model_validator(mode="after")
    def _bounded_copies(self) -> "NestRequest":
        copies = sum(i.quantity for i in self.items)
        if copies > MAX_PLACEMENTS:
            raise ValueError(f"{copies} copies to nest; at most {MAX_PLACEMENTS} per request")
        return self


@app.post("/nest")
def nest(req: NestRequest) -> dict:
    sheets = nesting.nest(
        [nesting.Item(i.id, i.width_in, i.height_in, i.quantity) for i in req.items],
        sheet_width_in=req.sheet_width_in,
        spacing_in=req.spacing_in,
        margin_in=req.margin_in,
        max_length_in=req.max_length_in,
        allow_rotation=req.allow_rotation,
        label_height_in=req.label_height_in,
        header_height_in=req.header_height_in,
    )
    return {
        "sheets": [
            {
                "index": s.index,
                "length_in": s.length_in,
                "utilization": s.utilization,
                "placements": [p.__dict__ for p in s.placements],
            }
            for s in sheets
        ]
    }


# --- compose ------------------------------------------------------------------------------


class LabelModel(BaseModel):
    order_no: str = Field("", max_length=100)
    item_no: str = Field("", max_length=100)
    size: str = Field("", max_length=100)
    color: str = Field("", max_length=100)
    design: str = Field("", max_length=200)
    reprint: bool = False


class ComposePlacementModel(BaseModel):
    transfer_id: str = Field(min_length=1, max_length=100)
    file_key: str = Key
    x_in: float = Field(ge=0, le=240)
    y_in: float = Field(ge=0, le=240)
    width_in: float = Field(gt=0, le=240)
    height_in: float = Field(gt=0, le=240)
    rotated: bool = False
    label: LabelModel | None = None


class ComposeRequest(BaseModel):
    width_in: float = Field(gt=0, le=60)
    length_in: float = Field(gt=0, le=240)
    dpi: int = Field(300, ge=36, le=1200)
    placements: list[ComposePlacementModel] = Field(max_length=MAX_PLACEMENTS)
    label_height_in: float = Field(0.42, ge=0, le=6)
    out_key: str = Key
    pdf_key: str | None = Field(None, min_length=1, max_length=512)
    preview_key: str = Key
    preview_width_px: int = Field(1200, ge=16, le=8000)
    label_gap_in: float = Field(compose.DEFAULT_LABEL_GAP_IN, ge=0, le=1)
    filename_hint: str | None = Field(None, max_length=150)
    # Header QR payload (gate finding A): the short sheet id, so the QR always fits the band.
    sheet_id: str | None = Field(None, min_length=1, max_length=64)
    header_height_in: float = Field(0.45, ge=0, le=6)

    @model_validator(mode="after")
    def _pdf_length_cap(self) -> "ComposeRequest":
        # B-80: PDF pages can't rely on /UserUnit (app/pdf.py), so cap PDF output at 200in here
        # too — a clear 422 before compositing, rather than after, if a caller bypasses the
        # SheetSpec cap enforced at spec save.
        if self.pdf_key is not None and self.length_in > PDF_MAX_LENGTH_IN:
            raise ValueError(
                f"PDF sheets are capped at {PDF_MAX_LENGTH_IN}in; use PNG for longer runs."
            )
        return self


@app.post("/compose")
def compose_sheet(req: ComposeRequest) -> dict:
    res = compose.compose_sheet(
        _CheckedStorage(get_storage()),
        width_in=req.width_in,
        length_in=req.length_in,
        dpi=req.dpi,
        placements=[
            compose.ComposePlacement(
                transfer_id=p.transfer_id,
                file_key=p.file_key,
                x_in=p.x_in,
                y_in=p.y_in,
                width_in=p.width_in,
                height_in=p.height_in,
                rotated=p.rotated,
                label=compose.Label(**p.label.model_dump()) if p.label else None,
            )
            for p in req.placements
        ],
        label_height_in=req.label_height_in,
        out_key=req.out_key,
        pdf_key=req.pdf_key,
        preview_key=req.preview_key,
        preview_width_px=req.preview_width_px,
        label_gap_in=req.label_gap_in,
        filename_hint=req.filename_hint,
        sheet_id=req.sheet_id,
        header_height_in=req.header_height_in,
    )
    out = {
        "key": req.out_key,
        "preview_key": req.preview_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "bytes": res.bytes,
        "render_ms": round(res.seconds * 1000),
    }
    if req.pdf_key:
        out["pdf_key"] = req.pdf_key
    return out


# --- mockup -------------------------------------------------------------------------------


class MockupRequest(BaseModel):
    design_key: str = Key
    blank_color_hex: str = Field(pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
    placement: Literal["front", "back"] = "front"
    out_key: str = Key
    size_px: int = Field(1200, ge=128, le=4000)


@app.post("/mockup")
def mockup(req: MockupRequest) -> dict:
    with workdir() as tmp:
        design = _download(req.design_key, tmp / "design")
        im = mockups.render_mockup(design, req.blank_color_hex, req.placement, req.size_px)
        out = tmp / "mockup.png"
        im.save(out, optimize=False, compress_level=6)
        get_storage().upload(out, req.out_key)
    return {"key": req.out_key}


# --- shipping label -----------------------------------------------------------------------


class AddressModel(BaseModel):
    name: str
    city: str
    state: str
    zip: str


class MockLabelRequest(BaseModel):
    shipment_id: str
    carrier: str
    service: str
    tracking_code: str = Field(min_length=1, max_length=64)
    from_: AddressModel = Field(alias="from")
    to: AddressModel
    weight_oz: float = Field(gt=0)
    out_key: str


@app.post("/labels/mock", dependencies=[Depends(_dev_only)])
def label_mock(req: MockLabelRequest) -> dict:
    pdf = labels.render_label(
        req.shipment_id,
        req.carrier,
        req.service,
        req.tracking_code,
        labels.Address(**req.from_.model_dump()),
        labels.Address(**req.to.model_dump()),
        req.weight_oz,
    )
    get_storage().put_bytes(pdf, req.out_key)
    return {"key": req.out_key}


class QrLabelItemModel(BaseModel):
    code: str = Field(min_length=1, max_length=200)
    caption: str | None = None
    size: Literal["4x6", "2x1"]


class QrLabelsRequest(BaseModel):
    labels: list[QrLabelItemModel] = Field(min_length=1, max_length=200)
    out_key: str = Key


@app.post("/labels/qr")
def label_qr(req: QrLabelsRequest) -> dict:
    """Merged PDF of `BIN:`/`B:` QR labels for production.bins.labels and inventory.blankLabels."""
    pdf = labels.render_qr_labels(
        [labels.QrLabelItem(code=i.code, caption=i.caption, size=i.size) for i in req.labels]
    )
    get_storage().put_bytes(pdf, req.out_key)
    return {"key": req.out_key}


# --- sample art ---------------------------------------------------------------------------


class SampleArtRequest(BaseModel):
    text: str = Field(min_length=1, max_length=120)
    out_key: str
    width_in: float = Field(gt=0, le=30)
    height_in: float = Field(gt=0, le=30)
    dpi: int = Field(300, ge=36, le=600)
    color_hex: str = Field("#111111", pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@app.post("/sample-art", dependencies=[Depends(_dev_only)])
def sample_art_endpoint(req: SampleArtRequest) -> dict:
    w, h = round(req.width_in * req.dpi), round(req.height_in * req.dpi)
    im = sample_art.render_sample_art(req.text, w, h, req.color_hex)
    with workdir() as tmp:
        out = tmp / "art.png"
        im.save(out, dpi=(req.dpi, req.dpi), compress_level=6)
        get_storage().upload(out, req.out_key)
    return {"key": req.out_key}
