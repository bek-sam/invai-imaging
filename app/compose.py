"""Compose a gang sheet with pyvips.

A 22in x 240in sheet at 300 DPI is 6,600 x 72,000 px (1.9 GB as RGBA), so nothing holds it in
memory: every design is resized/rotated once into a small temp PNG, the sheet is a lazy
`black` canvas with one `insert` per design and label, and `pngsave` pulls it through in strips
top to bottom. Designs are opened with sequential access, which matches that order. The PDF and
the preview are then streamed from the saved PNG.

Under each design (inside the label strip that nesting reserved) we print a QR code with the
transfer_id and "order_no · item_no · size · color · design", plus REPRINT when flagged.
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
from app.vips import ImageError, load, save_png, set_dpi, to_srgba, workdir

LABEL_GAP_IN = 0.04  # clear film between the design and its label
INK = (17, 17, 17, 255)
REPRINT_RED = (220, 38, 38, 255)

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


def qr_image(data: str, size_px: int) -> Image.Image:
    """Opaque white QR so it scans on clear film.

    The label strip is only ~0.3in tall, so the quiet zone is a single module: a UUID then fits
    at 3px (0.25mm) per module at 300 DPI instead of 2px with the usual wider border.
    """
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    n = len(matrix)
    small = Image.new("L", (n, n), 255)
    small.putdata([0 if cell else 255 for row in matrix for cell in row])
    # Whole pixels per module keep the modules crisp; pad the rest with white.
    module = max(1, size_px // n)
    big = small.resize((n * module, n * module), Image.Resampling.NEAREST)
    out = Image.new("RGBA", (size_px, size_px), (255, 255, 255, 255))
    off = (size_px - big.width) // 2
    out.paste(big.convert("RGBA"), (off, off))
    return out


def render_label(transfer_id: str, label: Label | None, width_px: int, height_px: int):
    im = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    qr_side = min(height_px, width_px)
    im.paste(qr_image(transfer_id, qr_side), (0, 0))
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

            label_h = round((label_height_in - LABEL_GAP_IN) * dpi)
            label_y = y + h + round(LABEL_GAP_IN * dpi)
            if label_h >= 8:
                label_h = min(label_h, H - label_y)
                if label_h >= 8:
                    lpath = tmp / f"label-{i}.png"
                    render_label(p.transfer_id, p.label, w, label_h).save(lpath, compress_level=1)
                    layers.append((lpath, x, label_y))

        sheet = pyvips.Image.black(W, H, bands=4).copy(interpretation="srgb")
        for src, x, y in layers:
            layer = to_srgba(pyvips.Image.new_from_file(str(src), access="sequential"))
            sheet = sheet.insert(layer, x, y)

        png = tmp / "sheet.png"
        save_png(set_dpi(sheet, dpi), png)
        del sheet
        storage.upload(png, out_key)

        if pdf_key:
            pdf = tmp / "sheet.pdf"
            write_image_pdf(png, pdf, W / dpi, H / dpi)
            storage.upload(pdf, pdf_key)

        preview = pyvips.Image.thumbnail(str(png), preview_width_px, height=10_000_000, size="down")
        prev_path = tmp / "preview.png"
        save_png(preview, prev_path)
        storage.upload(prev_path, preview_key)

        return ComposeResult(W, H, png.stat().st_size, time.monotonic() - started)
