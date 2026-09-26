"""Rasterize page-1 of an input PDF into a pyvips image at a target DPI.

libvips' binary wheel has no `pdfload` (checked live: `hasattr(pyvips.Image, "pdfload")` is
`False` on `pyvips[binary]==3.2.0` / libvips 8.18.6), so PDF *input* art can't go through
`pyvips.Image.new_from_file` like everything else. pypdfium2 ships a prebuilt pdfium with
pip-only wheels (no system Poppler needed, no Dockerfile change), so we rasterize with it and
hand the bitmap to pyvips via `new_from_memory`, so the rest of the pipeline (compose/QA/render)
sees a normal, in-memory vips image.

pdfium's native bitmap format is BGRA (confirmed live: `PdfBitmap.mode == "BGRA"`), so the R/B
bands are swapped before handing the buffer to callers that assume RGBA (`to_srgba`, `to_pil`).

Multi-page PDFs are rejected with a 422 (`ImageError`) — art uploads are single-page by contract.
"""

from pathlib import Path

import pypdfium2 as pdfium
import pyvips

from app.vips import ImageError

PT_PER_IN = 72.0


def rasterize_pdf(path: Path, dpi: float) -> pyvips.Image:
    try:
        doc = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        raise ImageError(f"unreadable pdf: {str(e).splitlines()[0]}") from e

    try:
        if len(doc) != 1:
            raise ImageError(f"pdf art must be single-page, got {len(doc)} pages")

        page = doc[0]
        scale = dpi / PT_PER_IN
        bitmap = page.render(scale=scale, draw_annots=False, fill_color=(255, 255, 255, 0))
        try:
            if bitmap.mode not in ("BGRA", "RGBA"):
                raise ImageError(f"unsupported pdf bitmap format: {bitmap.mode}")
            w, h, stride, n = bitmap.width, bitmap.height, bitmap.stride, bitmap.n_channels
            buf = bytes(bitmap.buffer)
        finally:
            bitmap.close()

        if stride == w * n:
            img = pyvips.Image.new_from_memory(buf, w, h, n, "uchar")
        else:
            # pdfium pads rows to a wider stride; load the padded width, then crop it off.
            img = pyvips.Image.new_from_memory(buf, stride // n, h, n, "uchar").crop(0, 0, w, h)

        if bitmap.mode == "BGRA":
            b, g, r, a = (img.extract_band(i) for i in range(4))
            img = r.bandjoin([g, b, a])

        # Force the pipeline into a private memory area now, before `buf` goes out of scope.
        return img.copy_memory().copy(interpretation="srgb")
    except pyvips.Error as e:
        raise ImageError(f"unreadable pdf: {str(e).splitlines()[0]}") from e
    finally:
        doc.close()
