"""Compose a gang sheet with pyvips.

A 22in x 240in sheet at 300 DPI is 6,600 x 72,000 px (1.9 GB as RGBA), so nothing holds it in
memory: every design is resized/rotated once into a small temp PNG, the sheet is a lazy
`black` canvas with one `insert` per design and label, and `pngsave` pulls it through in strips
top to bottom. Designs are opened with sequential access, which matches that order. The PDF and
the preview are then streamed from the saved PNG.

Under each design (inside the label strip that nesting reserved) we print a QR code with the
transfer_id and "order_no · item_no · size · color · design", plus REPRINT when flagged. A thin
dashed cut guide marks `label_gap_in` (the clear film between the design and its label), and an
optional header strip at the top of the sheet carries a second QR (plus text) for `filename_hint`
so a RIP or a human can look the job up by name or by scanning the code (B-79).

Every transfer-label QR is drawn from a raster of at least `QR_MIN_DPI`, whatever the sheet's own
`dpi` is, and every module is held to at least `MIN_MODULE_MM` -- when the label strip is too
short for that, compose fails loudly (`ImageError`) instead of printing an unscannable code.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pyvips
import qrcode
from PIL import Image, ImageDraw

from app.fonts import font, text_size
from app.pdf import write_image_pdf
from app.storage import NotFoundError, Storage
from app.vips import MM_PER_IN, ImageError, load, save_png, set_dpi, to_srgba, workdir

DEFAULT_LABEL_GAP_IN = 0.125  # clear film between the design and its label (SheetSpec.labelGapIn)
MIN_MODULE_MM = 0.5  # AC1: every QR module must be at least this big, printed
QR_MIN_DPI = 300  # AC1: transfer-label QRs are drawn from at least this raster density
HEADER_MAX_IN = 1.0  # header strip cap, even if the sheet's top margin is unusually generous
INK = (17, 17, 17, 255)
REPRINT_RED = (220, 38, 38, 255)
CUT_GUIDE_COLOR = (120, 120, 120, 200)

# The same design file can be inserted many times; with the operation cache on, pyvips would hand
# back one shared sequential loader for all of them and fail on out-of-order reads.
pyvips.cache_set_max(0)


@dataclass
class Label:
    order_no: str = ""
    item_no: str = ""
    size: str = ""
    color: str = ""
    design: str = ""
    reprint: bool = False


@dataclass
class ComposePlacement:
    transfer_id: str
    file_key: str
    x_in: float
    y_in: float
    width_in: float
    height_in: float
    rotated: bool
    label: Label | None


@dataclass
class ComposeResult:
    width_px: int
    height_px: int
    bytes: int
    seconds: float


def label_text(label: Label) -> str:
    parts = [label.order_no, label.item_no, label.size, label.color, label.design]
    return " · ".join(p for p in parts if p)


def _qr_matrix(data: str) -> list[list[bool]]:
    # The label strip is short, so the quiet zone is compressed to a single module.
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.get_matrix()


def qr_min_side_in(data: str) -> float:
    """Minimum physical QR side, in inches, so every module of `data`'s code is >= MIN_MODULE_MM."""
    return len(_qr_matrix(data)) * MIN_MODULE_MM / MM_PER_IN


def _qr_bitmap(matrix: list[list[bool]], avail_px: int, dpi: float) -> Image.Image:
    """Render `matrix` filling an `avail_px` square, from a raster of at least QR_MIN_DPI.

    Building the master at max(dpi, QR_MIN_DPI) and resampling down (not up) to `avail_px` means
    a sheet composed at a low DPI never gets blockier modules than one at 300+: the master is
    always at least as dense.
    """
    n = len(matrix)
    master_module_px = max(1, round(MIN_MODULE_MM / MM_PER_IN * max(dpi, QR_MIN_DPI)))
    master_side = n * master_module_px
    small = Image.new("L", (n, n), 255)
    small.putdata([0 if cell else 255 for row in matrix for cell in row])
    resample = Image.Resampling.NEAREST if avail_px >= master_side else Image.Resampling.LANCZOS
    big = small.resize((avail_px, avail_px), resample)
    out = Image.new("RGBA", (avail_px, avail_px), (255, 255, 255, 255))
    out.paste(big.convert("RGBA"), (0, 0))
    return out


def qr_image(data: str, avail_px: int, dpi: float) -> Image.Image:
    """Opaque white transfer-label QR so it scans on clear film (AC1).

    `avail_px` is the label strip's available square at the sheet's own `dpi`. Raises
    `ImageError` -- compose fails loudly -- when that square can't hold every module of `data`'s
    QR at >= MIN_MODULE_MM: shrinking further would make the code unscannable.
    """
    matrix = _qr_matrix(data)
    n = len(matrix)
    min_side_in = n * MIN_MODULE_MM / MM_PER_IN
    avail_in = avail_px / dpi
    if avail_in < min_side_in:
        raise ImageError(
            f"label strip too short for a scannable QR: {n} modules need >= "
            f"{min_side_in:.3f}in at {MIN_MODULE_MM}mm/module, only {avail_in:.3f}in "
            f"available at {dpi} DPI"
        )
    return _qr_bitmap(matrix, avail_px, dpi)


def header_qr_image(data: str, avail_px: int, dpi: float) -> Image.Image | None:
    """Same construction as `qr_image`, but returns None instead of raising: a header that can't
    fit a legible QR is dropped, it never fails the whole compose (unlike a transfer label)."""
    matrix = _qr_matrix(data)
    if avail_px <= 0 or avail_px / dpi < len(matrix) * MIN_MODULE_MM / MM_PER_IN:
        return None
    return _qr_bitmap(matrix, avail_px, dpi)


def render_cut_guide(width_px: int, thickness_px: int) -> Image.Image:
    """A thin dashed line marking `label_gap_in`, the clear film between a design and its label."""
    im = Image.new("RGBA", (max(1, width_px), max(1, thickness_px)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    dash, gap = max(4, thickness_px * 6), max(3, thickness_px * 4)
    x = 0
    while x < width_px:
        draw.rectangle([x, 0, min(x + dash, width_px) - 1, thickness_px - 1], fill=CUT_GUIDE_COLOR)
        x += dash + gap
    return im


def render_label(transfer_id: str, label: Label | None, width_px: int, height_px: int, dpi: float):
    im = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    qr_side = min(height_px, width_px)
    im.paste(qr_image(transfer_id, qr_side, dpi), (0, 0))
    avail = width_px - qr_side - max(4, height_px // 8)
    if label is None or avail < height_px // 2:
        return im
    x = qr_side + max(4, height_px // 8)
    draw = ImageDraw.Draw(im)
    text = label_text(label)
    reprint = "REPRINT" if label.reprint else ""
    # Two lines when there is room for them: the label text, and REPRINT on its own line.
    size = int(height_px * (0.42 if reprint else 0.55))
    min_size = max(8, height_px // 5)
    while size > min_size:
        f = font("Inter Bold", size)
        if text_size(f, text)[0] <= avail and text_size(f, reprint)[0] <= avail:
            break
        size -= 1
    f = font("Inter Bold", size)
    if text_size(f, text)[0] > avail:
        while text and text_size(f, text + "…")[0] > avail:
            text = text[:-1]
        text = text + "…" if text else ""
    if reprint:
        rf = font("Inter Black", size)
        draw.text((x, 0), reprint, font=rf, fill=REPRINT_RED, anchor="lt")
        draw.text((x, height_px), text, font=f, fill=INK, anchor="lb")
    else:
        draw.text((x, height_px // 2), text, font=f, fill=INK, anchor="lm")
    return im


def render_header(sheet_label: str, width_px: int, height_px: int, dpi: float) -> Image.Image:
    """Sheet header strip (AC2): a QR encoding `sheet_label` plus the same text, so CADlink (or
    a human) can look the job up by scanning the code or by its name -- both carry the sheet id.
    Dropped silently (returns a blank strip) if the header band is too short for a legible QR.
    """
    im = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    qr_side = min(height_px, width_px)
    qr = header_qr_image(sheet_label, qr_side, dpi)
    if qr is None:
        return im
    im.paste(qr, (0, 0))
    avail = width_px - qr_side - max(4, height_px // 8)
    if avail < height_px // 2:
        return im
    x = qr_side + max(4, height_px // 8)
    draw = ImageDraw.Draw(im)
    size = int(height_px * 0.6)
    min_size = max(8, height_px // 4)
    f = font("Inter Bold", size)
    while size > min_size and text_size(f, sheet_label)[0] > avail:
        size -= 1
        f = font("Inter Bold", size)
    text = sheet_label
    if text_size(f, text)[0] > avail:
        while text and text_size(f, text + "…")[0] > avail:
            text = text[:-1]
        text = text + "…" if text else ""
    draw.text((x, height_px // 2), text, font=f, fill=INK, anchor="lm")
    return im


def _prepare_design(src: Path, dest: Path, w: int, h: int, rotated: bool) -> tuple[int, int]:
    """Resize (and rotate) one design to fit w x h px; returns the result size."""
    img = load(src)
    sw, sh = (img.height, img.width) if rotated else (img.width, img.height)
    scale = min(w / sw, h / sh)
    img = to_srgba(img)
    # Resize in premultiplied space so transparent edges don't pick up dark fringes.
    img = img.premultiply().resize(scale, kernel="lanczos3").unpremultiply().cast("uchar")
    if rotated:
        img = img.rot90()
    if img.width > w or img.height > h:
        img = img.crop(0, 0, min(img.width, w), min(img.height, h))
    save_png(img, dest, compression=1)
    return img.width, img.height


def compose_sheet(
    storage: Storage,
    width_in: float,
    length_in: float,
    dpi: int,
    placements: list[ComposePlacement],
    label_height_in: float,
    out_key: str,
    pdf_key: str | None,
    preview_key: str,
    preview_width_px: int,
    label_gap_in: float = DEFAULT_LABEL_GAP_IN,
    filename_hint: str | None = None,
) -> ComposeResult:
    started = time.monotonic()
    W, H = round(width_in * dpi), round(length_in * dpi)
    if W <= 0 or H <= 0:
        raise ImageError("width_in and length_in must be positive")
    if W * H > 50_000 * 100_000:
        raise ImageError("sheet is too large")

    with workdir() as tmp:
        sources: dict[str, Path] = {}
        variants: dict[tuple, tuple[Path, int, int]] = {}
        layers: list[tuple[Path, int, int]] = []

        if filename_hint:
            # Fit inside whatever clearance the caller already left above the first placement
            # (the sheet's top margin) -- never a fixed height that could bleed into a design.
            top_clear_px = min((round(p.y_in * dpi) for p in placements), default=H)
            header_h = min(top_clear_px, round(HEADER_MAX_IN * dpi), H)
            if header_h >= 8:
                hpath = tmp / "header.png"
                render_header(filename_hint, W, header_h, dpi).save(hpath, compress_level=1)
                layers.append((hpath, 0, 0))

        for i, p in enumerate(placements):
            x, y = round(p.x_in * dpi), round(p.y_in * dpi)
            w, h = round(p.width_in * dpi), round(p.height_in * dpi)
            if w <= 0 or h <= 0:
                raise ImageError(f"placement {i} ({p.transfer_id}) has no size")
            if x < 0 or y < 0 or x + w > W or y + h > H:
                raise ImageError(f"placement {i} ({p.transfer_id}) is outside the sheet")

            if p.file_key not in sources:
                path = tmp / f"src-{len(sources)}"
                try:
                    storage.download(p.file_key, path)
                except NotFoundError as e:
                    raise ImageError(f"file not found: {p.file_key}") from e
                sources[p.file_key] = path
            vkey = (p.file_key, w, h, p.rotated)
            if vkey not in variants:
                vpath = tmp / f"design-{len(variants)}.png"
                vw, vh = _prepare_design(sources[p.file_key], vpath, w, h, p.rotated)
                variants[vkey] = (vpath, vw, vh)
            vpath, vw, vh = variants[vkey]
            layers.append((vpath, x + (w - vw) // 2, y + (h - vh) // 2))

            gap_px = round(label_gap_in * dpi)
            label_h = round((label_height_in - label_gap_in) * dpi)
            label_y = y + h + gap_px
            guide_h = max(1, round(0.006 * dpi))
            if gap_px > guide_h + 1:
                guide_y = y + h + (gap_px - guide_h) // 2
                if guide_y + guide_h <= H:
                    gpath = tmp / f"guide-{i}.png"
                    render_cut_guide(w, guide_h).save(gpath, compress_level=1)
                    layers.append((gpath, x, guide_y))
            if label_h >= 8:
                label_h = min(label_h, H - label_y)
                if label_h >= 8:
                    lpath = tmp / f"label-{i}.png"
                    render_label(p.transfer_id, p.label, w, label_h, dpi).save(
                        lpath, compress_level=1
                    )
                    layers.append((lpath, x, label_y))

        sheet = pyvips.Image.black(W, H, bands=4).copy(interpretation="srgb")
        for src, x, y in layers:
            layer = to_srgba(pyvips.Image.new_from_file(str(src), access="sequential"))
            sheet = sheet.insert(layer, x, y)

        png = tmp / "sheet.png"
        save_png(set_dpi(sheet, dpi), png)
        del sheet
        storage.upload(png, out_key, f"{filename_hint}.png" if filename_hint else None)

        if pdf_key:
            pdf = tmp / "sheet.pdf"
            write_image_pdf(png, pdf, W / dpi, H / dpi)
            storage.upload(pdf, pdf_key, f"{filename_hint}.pdf" if filename_hint else None)

        preview = pyvips.Image.thumbnail(str(png), preview_width_px, height=10_000_000, size="down")
        prev_path = tmp / "preview.png"
        save_png(preview, prev_path)
        storage.upload(prev_path, preview_key)

        return ComposeResult(W, H, png.stat().st_size, time.monotonic() - started)
