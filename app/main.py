"""invai-imaging HTTP API (v1-plan.md section 5.1). Inputs and outputs are storage keys."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pyvips
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import compose, labels, mockups, nesting, qa, render, sample_art
from app.config import settings
from app.storage import NotFoundError, S3Storage, get_storage
from app.vips import ImageError, workdir

log = logging.getLogger("invai.imaging")


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage = get_storage()
    if isinstance(storage, S3Storage) and settings.s3_create_bucket:
        try:
            storage.ensure_bucket()
        except (BotoCoreError, ClientError) as e:  # keep serving; requests report the error
            log.warning("could not ensure bucket %s: %s", settings.s3_bucket, e)
    yield


app = FastAPI(title="invai-imaging", lifespan=lifespan)


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


def _download(key: str, dest: Path) -> Path:
    get_storage().download(key, dest)
    return dest


# --- health -------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True, "vips_version": ".".join(str(pyvips.version(i)) for i in range(3))}


# --- QA -----------------------------------------------------------------------------------


class QACheckRequest(BaseModel):
    file_key: str
    target_width_in: float | None = Field(None, gt=0)
    target_height_in: float | None = Field(None, gt=0)


@app.post("/qa/check")
def qa_check(req: QACheckRequest) -> dict:
    with workdir() as tmp:
        path = _download(req.file_key, tmp / "input")
        return qa.check(path, req.target_width_in, req.target_height_in).to_dict()


class CleanAlphaRequest(BaseModel):
    file_key: str
    out_key: str
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
    name: str
    kind: Literal["text"] = "text"
    x_in: float = Field(ge=0)
    y_in: float = Field(ge=0)
    w_in: float = Field(gt=0)
    h_in: float = Field(gt=0)
    font_family: str = "Inter"
    font_size_pt: float = Field(36, gt=0)
    color: str = "#000000"
    align: Literal["left", "center", "right"] = "center"
    max_chars: int | None = Field(None, gt=0)
    uppercase: bool = False


class TemplateModel(BaseModel):
    width_in: float = Field(gt=0, le=60)
    height_in: float = Field(gt=0, le=60)
    background_key: str | None = None
    slots: list[SlotModel]


class PersonalizationRequest(BaseModel):
    template: TemplateModel
    values: dict[str, str] = {}
    out_key: str
    dpi: int = Field(300, ge=36, le=1200)


@app.post("/render/personalization")
def render_personalization(req: PersonalizationRequest) -> dict:
    t = req.template
    slots = [render.Slot(**s.model_dump()) for s in t.slots]
    with workdir() as tmp:
        bg = _download(t.background_key, tmp / "background") if t.background_key else None
        out = tmp / "out.png"
        res = render.render_personalization(
            t.width_in, t.height_in, slots, req.values, req.dpi, out, bg
        )
        get_storage().upload(out, req.out_key)
    return {
        "key": req.out_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "flags": [f.__dict__ for f in res.flags],
    }


# --- nesting ------------------------------------------------------------------------------


class NestItemModel(BaseModel):
    id: str
    width_in: float = Field(gt=0)
    height_in: float = Field(gt=0)
    quantity: int = Field(1, ge=1, le=10_000)


class NestRequest(BaseModel):
    items: list[NestItemModel]
    sheet_width_in: float = Field(22.0, gt=0)
    spacing_in: float = Field(0.25, ge=0)
    margin_in: float = Field(0.25, ge=0)
    max_length_in: float = Field(240.0, gt=0)
    allow_rotation: bool = True
    label_height_in: float = Field(0.35, ge=0)


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
    order_no: str = ""
    item_no: str = ""
    size: str = ""
    color: str = ""
    design: str = ""
    reprint: bool = False


class ComposePlacementModel(BaseModel):
    transfer_id: str
    file_key: str
    x_in: float
    y_in: float
    width_in: float = Field(gt=0)
    height_in: float = Field(gt=0)
    rotated: bool = False
    label: LabelModel | None = None


class ComposeRequest(BaseModel):
    width_in: float = Field(gt=0)
    length_in: float = Field(gt=0)
    dpi: int = Field(300, ge=36, le=1200)
    placements: list[ComposePlacementModel]
    label_height_in: float = Field(0.35, ge=0)
    out_key: str
    pdf_key: str | None = None
    preview_key: str
    preview_width_px: int = Field(1200, ge=16, le=8000)


@app.post("/compose")
def compose_sheet(req: ComposeRequest) -> dict:
    res = compose.compose_sheet(
        get_storage(),
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
    )
    out = {
        "key": req.out_key,
        "preview_key": req.preview_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "bytes": res.bytes,
    }
    if req.pdf_key:
        out["pdf_key"] = req.pdf_key
    return out


# --- mockup -------------------------------------------------------------------------------


class MockupRequest(BaseModel):
    design_key: str
    blank_color_hex: str = Field(pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
    placement: Literal["front", "back"] = "front"
    out_key: str
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


@app.post("/labels/mock")
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
    out_key: str


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


@app.post("/sample-art")
def sample_art_endpoint(req: SampleArtRequest) -> dict:
    w, h = round(req.width_in * req.dpi), round(req.height_in * req.dpi)
    im = sample_art.render_sample_art(req.text, w, h, req.color_hex)
    with workdir() as tmp:
        out = tmp / "art.png"
        im.save(out, dpi=(req.dpi, req.dpi), compress_level=6)
        get_storage().upload(out, req.out_key)
    return {"key": req.out_key}
