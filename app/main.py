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

from app import (
    compose,
    fonts,
    garments,
    labels,
    mockups,
    nesting,
    photos,
    qa,
    render,
    sample_art,
    scenes,
)
from app.config import production_config_error, settings
from app.guard import GuardMiddleware, limiter
from app.keys import KeyScopeError, check_scope, company_of
from app.limits import declared_type_error
from app.pdf import PDF_MAX_LENGTH_IN
from app.storage import NotFoundError, S3Storage, Storage, get_storage
from app.telemetry import setup_tracing
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
# Tracing only when OTEL_EXPORTER_OTLP_ENDPOINT is set (app/telemetry.py, T-32-5).
setup_tracing(app)

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


@app.exception_handler(KeyScopeError)
async def key_scope_error(_: Request, exc: KeyScopeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(scenes.SceneInputError)
async def scene_input_error(_: Request, exc: scenes.SceneInputError):
    # The provider's scene is untrusted model output: refused as a bad request (card T-27-2).
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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


# B-103 added `aspect_mismatch` and `upscale` photo-flag codes. The backend client
# (invai-backend/src/integrations/imaging/client.ts) parses `flags[].code` with a closed enum
# that doesn't know them yet, so they're opt-in: a caller passes `photo_flags: true` to get them.
# Default response stays byte-for-byte what it was before this card.
_NEW_PHOTO_FLAG_CODES = {"aspect_mismatch", "upscale"}


class PersonalizationRequest(BaseModel):
    template: TemplateModel
    values: dict[str, str] = Field(default_factory=dict, max_length=50)
    out_key: str = Key
    dpi: int = Field(300, ge=36, le=1200)
    photo_flags: bool = False

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
    flags = res.flags
    if not req.photo_flags:
        flags = [f for f in flags if f.code not in _NEW_PHOTO_FLAG_CODES]
    return {
        "key": req.out_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "flags": [f.__dict__ for f in flags],
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
        ],
        # B-41: length-weighted film use across the whole sheet set, alongside each sheet's own
        # `utilization` above -- additive field, existing callers reading `sheets` are unaffected.
        "overall_utilization": nesting.overall_utilization(sheets),
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
    # B-103: some DTF vendors' press/RIP expects a pre-mirrored print. Flips only the design
    # artwork, never the label strip or header, so a presser can still read and scan them.
    mirror: bool = False

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
        mirror=req.mirror,
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


# --- listing photos (T-26-2) ----------------------------------------------------------------

Garment = Literal["tee", "hoodie", "crewneck", "tank"]
PhotoView = Literal["front_flat", "folded", "back", "on_model_white"]
PhotoPreset = Literal["amazon_main", "amazon_alt", "etsy", "shopify", "tiktok", "walmart"]


class PhotoPaletteRequest(BaseModel):
    design_key: str = Key


@app.post("/photo/palette")
def photo_palette(req: PhotoPaletteRequest) -> dict:
    company_of(req.design_key, "design_key")
    with workdir() as tmp:
        return photos.palette(_download(req.design_key, tmp / "design"))


@app.get("/photo/templates")
def photo_templates() -> list[dict]:
    out = []
    for t in garments.all_templates():
        a = t.print_area
        out.append(
            {
                "garment": t.garment,
                "view": t.view,
                "placement": t.placement,
                "print_areas": {t.placement: {"w_in": a.w, "h_in": a.h}},
                "drawn": t.drawn,
            }
        )
    return out


class PhotoRenderRequest(BaseModel):
    design_key: str = Key
    design_width_in: float = Field(gt=0, le=60)
    design_height_in: float = Field(gt=0, le=60)
    placement: Literal["front", "back"] = "front"
    garment: Garment
    view: PhotoView
    blank_hex: str = Field(pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
    underbase_preview: bool = False
    preset: PhotoPreset
    out_key: str = Key
    xmp_subjects: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("xmp_subjects")
    @classmethod
    def _subjects(cls, v: list[str]) -> list[str]:
        for s in v:
            if not 1 <= len(s) <= 100:
                raise ValueError("each XMP subject is 1-100 characters")
        return v

    @model_validator(mode="after")
    def _placement_matches_view(self) -> "PhotoRenderRequest":
        wanted = "back" if self.view == "back" else "front"
        if self.placement != wanted:
            raise ValueError(f"view {self.view} shows the {wanted} print; use placement {wanted}")
        return self


@app.post("/photo/render")
def photo_render(req: PhotoRenderRequest) -> dict:
    check_scope(req.out_key, {"design_key": req.design_key})
    fmt = photos.output_format(req.out_key)
    started = time.perf_counter()
    with workdir() as tmp:
        design = _download(req.design_key, tmp / "design")
        res = photos.render_photo(
            design,
            req.design_width_in,
            req.design_height_in,
            req.garment,
            req.view,
            req.blank_hex,
            req.underbase_preview,
            req.preset,
            fmt,
            req.xmp_subjects,
        )
        get_storage().put_bytes(res.data, req.out_key)
    c = res.checks
    return {
        "key": req.out_key,
        "width_px": res.width_px,
        "height_px": res.height_px,
        "format": res.format,
        "print_box_px": list(res.print_box_px),
        "checks": {
            "passes": c.passes,
            "failures": c.failures,
            "background_pure_white": c.background_pure_white,
            "fill_ratio": c.fill_ratio,
            "longest_side_px": c.longest_side_px,
        },
        # Additive extras: the template's print area, the scale applied to the design (1 = real
        # size) and whether the white underbase preview was drawn.
        "print_area_px": list(res.print_area_px),
        "scale": round(res.scale, 4),
        "underbase_applied": res.underbase_applied,
        "bytes": len(res.data),
        "render_ms": round((time.perf_counter() - started) * 1000),
    }


# --- AI lifestyle scenes (T-27-2, ADR 0023 phase B) -----------------------------------------

SceneView = Literal["on_model_white", "lifestyle_base"]
HexColor = Field(pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class SceneBaseRequest(BaseModel):
    garment: Garment
    view: SceneView
    blank_hex: str = HexColor
    # One number is a square base; [w, h] another aspect. Match the provider's output size
    # (GPT Image: 1024, [1536, 1024] or [1024, 1536]) so the scene comes back the same shape.
    size_px: int | list[int] = 1024
    out_key: str = Key
    mask_out_key: str = Key

    @field_validator("size_px")
    @classmethod
    def _size(cls, v: int | list[int]) -> int | list[int]:
        sides = [v] if isinstance(v, int) else v
        if len(sides) not in (1, 2) or any(not 256 <= s <= 2048 for s in sides):
            raise ValueError("size_px is one side or [w, h], each 256-2048")
        return v

    def wh(self) -> tuple[int, int]:
        v = self.size_px
        return (v, v) if isinstance(v, int) else (v[0], v[-1])


@app.post("/photo/scene-base")
def photo_scene_base(req: SceneBaseRequest) -> dict:
    check_scope(req.out_key, {"mask_out_key": req.mask_out_key})
    for what, key in (("out_key", req.out_key), ("mask_out_key", req.mask_out_key)):
        if not key.lower().endswith(".png"):
            raise ImageError(f"{what} must end in .png")
    if req.out_key == req.mask_out_key:
        raise ImageError("out_key and mask_out_key must differ")
    w, h = req.wh()
    res = scenes.render_base(req.garment, req.view, w, h, req.blank_hex)
    storage = get_storage()
    storage.put_bytes(res.png, req.out_key)
    storage.put_bytes(res.mask_png, req.mask_out_key)
    return {
        "key": req.out_key,
        "mask_key": req.mask_out_key,
        "print_box_px": list(res.print_box_px),
        # Additive extras: the base size, the protected (opaque) box in the mask, the template's
        # pixels per inch at this size.
        "width_px": res.width_px,
        "height_px": res.height_px,
        "protected_box_px": list(res.protected_box_px),
        "ppi": res.ppi,
    }


class SceneCompositeRequest(BaseModel):
    scene_key: str = Key
    base_key: str = Key
    print_box_px: list[int] = Field(min_length=4, max_length=4)
    design_key: str = Key
    design_width_in: float = Field(gt=0, le=60)
    design_height_in: float = Field(gt=0, le=60)
    placement: Literal["front"] = "front"
    blank_hex: str = HexColor
    preset: PhotoPreset
    out_key: str = Key
    xmp_subjects: list[str] = Field(default_factory=list, max_length=20)
    # Optional: the base records its garment and view; pass them only for a base made elsewhere.
    garment: Garment | None = None
    view: SceneView | None = None
    underbase_preview: bool = False

    @field_validator("xmp_subjects")
    @classmethod
    def _subjects(cls, v: list[str]) -> list[str]:
        return PhotoRenderRequest._subjects(v)


# The provider's scene is untrusted model output: checked from storage metadata before any byte
# is downloaded, then by magic bytes, header size and pixel cap before decode (400 when refused).
@app.post("/photo/scene-composite")
def photo_scene_composite(req: SceneCompositeRequest) -> dict:
    check_scope(
        req.out_key,
        {"scene_key": req.scene_key, "base_key": req.base_key, "design_key": req.design_key},
    )
    fmt = photos.output_format(req.out_key)
    storage = get_storage()
    scene_bytes = storage.size(req.scene_key)
    if scene_bytes > scenes.SCENE_MAX_BYTES:
        raise scenes.SceneInputError(
            f"scene is {scene_bytes / 1e6:.0f} MB, over the "
            f"{scenes.SCENE_MAX_BYTES / 1e6:.0f} MB limit"
        )
    started = time.perf_counter()
    with workdir() as tmp:
        scene_path = tmp / "scene"
        storage.download(req.scene_key, scene_path)
        if err := declared_type_error(req.scene_key, scene_path):
            raise scenes.SceneInputError(f"scene: {err}")
        res = scenes.composite_scene(
            scene_path,
            _download(req.base_key, tmp / "base"),
            tuple(req.print_box_px),
            _download(req.design_key, tmp / "design"),
            req.design_width_in,
            req.design_height_in,
            req.blank_hex,
            req.preset,
            fmt,
            req.xmp_subjects,
            underbase_preview=req.underbase_preview,
            garment=req.garment,
            view=req.view,
        )
        if res.data is not None:
            storage.put_bytes(res.data, req.out_key)
    return {
        # null when a check failed: nothing is written.
        "key": req.out_key if res.data is not None else None,
        "checks": {
            "passes": res.passes,
            "failures": res.failures,
            "region_unchanged_score": res.region_unchanged_score,
            "design_lock_score": res.design_lock_score,
            # Additive: the thresholds the scores were held to.
            "region_unchanged_min": scenes.REGION_UNCHANGED_MIN,
            "design_lock_min": scenes.DESIGN_LOCK_MIN,
            # Additive (T-37-1): which sub-checks of check 1 failed, the check-1b fraction, and
            # whether a regenerate with the same request can't fix it.
            "region_reasons": res.region_reasons,
            "box_foreign": res.box_foreign,
            "structural": res.structural,
            # Additive (T-49-5): CIEDE2000 of the restored blank vs the requested colour (null
            # when nothing could be measured), and warnings that never change `passes`.
            "blank_delta_e": res.blank_delta_e,
            "warnings": res.warnings,
        },
        # Additive extras.
        "width_px": res.width_px,
        "height_px": res.height_px,
        "format": res.format,
        "print_box_px": list(res.print_box_px),
        "preset_failures": res.preset_failures,
        "registration": res.registration,
        "upscale": round(res.upscale, 4),
        "upscale_method": "lbb",
        "underbase_applied": res.underbase_applied,
        "bytes": len(res.data) if res.data is not None else 0,
        "render_ms": round((time.perf_counter() - started) * 1000),
    }


class PhotoZipItem(BaseModel):
    key: str = Key
    name: str = Field(min_length=1, max_length=200)


class PhotoZipRequest(BaseModel):
    items: list[PhotoZipItem] = Field(min_length=1, max_length=500)
    out_key: str = Key


# Zip budget (S-50): up to 500 items with no cap let one call copy any kind of file in the
# company prefix (gang-sheet print files, CSV uploads) into imaging's temp disk and fill it,
# stopping compose, nest and render for every shop. These caps are checked from storage metadata
# -- before any bytes are downloaded -- and the allowed extensions keep `/photo/zip` to the
# photo outputs it exists for.
PHOTO_ZIP_MAX_ITEM_BYTES = 50 * 1024 * 1024  # 50 MB
PHOTO_ZIP_MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GB
PHOTO_ZIP_ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")


@app.post("/photo/zip")
def photo_zip(req: PhotoZipRequest) -> dict:
    check_scope(req.out_key, {f"items[{i}].key": it.key for i, it in enumerate(req.items)})
    if not req.out_key.lower().endswith(".zip"):
        raise ImageError("out_key must end in .zip")
    storage = get_storage()
    total_bytes = 0
    for i, it in enumerate(req.items):
        if not it.key.lower().endswith(PHOTO_ZIP_ALLOWED_EXTENSIONS):
            raise ImageError(f"items[{i}].key must be a .jpg, .jpeg or .png file")
        item_bytes = storage.size(it.key)
        if item_bytes > PHOTO_ZIP_MAX_ITEM_BYTES:
            raise ImageError(
                f"items[{i}].key is {item_bytes / 1e6:.0f} MB, over the "
                f"{PHOTO_ZIP_MAX_ITEM_BYTES / 1e6:.0f} MB per-item limit"
            )
        total_bytes += item_bytes
        if total_bytes > PHOTO_ZIP_MAX_TOTAL_BYTES:
            raise ImageError(
                f"items total {total_bytes / 1e6:.0f} MB, over the "
                f"{PHOTO_ZIP_MAX_TOTAL_BYTES / 1e6:.0f} MB total limit"
            )
    with workdir() as tmp:
        files = []
        checked = _CheckedStorage(storage)
        for i, it in enumerate(req.items):
            dest = tmp / f"item-{i}"
            checked.download(it.key, dest)
            files.append((dest, it.name))
        out = tmp / "photos.zip"
        size = photos.build_zip(files, out)
        storage.upload(out, req.out_key)
    return {"key": req.out_key, "bytes": size}


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
