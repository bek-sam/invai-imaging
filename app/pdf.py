"""Stream an RGBA raster into a single-page PDF at its true physical size.

libvips in the binary wheel has no pdfsave, and img2pdf/reportlab would decode the whole sheet into
memory (a 22 x 240in sheet is 1.9 GB of RGBA). So we write the PDF ourselves: the image is read in
strips, RGB and alpha are deflated into two temp streams (image XObject + SMask), then the file is
assembled.

B-80 decision (cap, not split): PDF pages beyond 200in (14,400pt, the classic Acrobat limit) used
to fall back to /UserUnit to keep the physical size correct, but several RIPs/viewers (including
some CADlink versions) don't honor /UserUnit, so a long sheet printed at the wrong size. Rather
than special-case that branch, PDF output is capped at 200in: `SheetSpec` rejects a PDF vendor
spec longer than that at save time (`invai-contracts/src/schemas/vendors.ts`, enforced again in
`invai-backend`'s vendor service), and this function raises defensively in case anything bypasses
that check. PNG has no page-size concept, so it keeps its own (higher) ceiling.
"""

import shutil
import zlib
from pathlib import Path

import pyvips

from app.vips import ImageError, to_srgba

STRIP_ROWS = 256
PDF_MAX_LENGTH_IN = 200
MAX_PAGE_PT = PDF_MAX_LENGTH_IN * 72


def write_image_pdf(src: Path, dest: Path, width_in: float, height_in: float) -> None:
    if max(width_in, height_in) > PDF_MAX_LENGTH_IN:
        raise ImageError(
            f"PDF sheets are capped at {PDF_MAX_LENGTH_IN}in; use PNG for longer runs."
        )
    img = to_srgba(pyvips.Image.new_from_file(str(src), access="sequential"))
    w, h = img.width, img.height
    rgb_path, alpha_path = dest.with_suffix(".rgb.z"), dest.with_suffix(".a.z")

    region = pyvips.Region.new(img)
    rgb_z = zlib.compressobj(6)
    a_z = zlib.compressobj(6)
    with open(rgb_path, "wb") as rgb_f, open(alpha_path, "wb") as a_f:
        for y in range(0, h, STRIP_ROWS):
            rows = min(STRIP_ROWS, h - y)
            data = bytes(region.fetch(0, y, w, rows))
            rgb = bytearray(w * rows * 3)
            rgb[0::3] = data[0::4]
            rgb[1::3] = data[1::4]
            rgb[2::3] = data[2::4]
            rgb_f.write(rgb_z.compress(rgb))
            a_f.write(a_z.compress(data[3::4]))
        rgb_f.write(rgb_z.flush())
        a_f.write(a_z.flush())

    # width_in/height_in are already <= PDF_MAX_LENGTH_IN (checked above), so page_w/page_h are
    # always <= MAX_PAGE_PT and no /UserUnit scaling is needed.
    page_w, page_h = width_in * 72, height_in * 72

    offsets: list[int] = []
    with open(dest, "wb") as out:

        def obj(body: bytes, stream_path: Path | None = None):
            offsets.append(out.tell())
            out.write(f"{len(offsets)} 0 obj\n".encode())
            out.write(body)
            if stream_path is not None:
                out.write(b"\nstream\n")
                with open(stream_path, "rb") as s:
                    shutil.copyfileobj(s, out, 1 << 20)
                out.write(b"\nendstream")
            out.write(b"\nendobj\n")

        out.write(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
        content = f"q {page_w:.4f} 0 0 {page_h:.4f} 0 0 cm /Im0 Do Q".encode()
        obj(b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        obj(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}]"
                f" /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            ).encode()
        )
        obj(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
        obj(
            (
                f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                f"/SMask 6 0 R /Length {rgb_path.stat().st_size} >>"
            ).encode(),
            rgb_path,
        )
        obj(
            (
                f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
                f"/Length {alpha_path.stat().st_size} >>"
            ).encode(),
            alpha_path,
        )
        xref = out.tell()
        out.write(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode())
        out.writelines(f"{off:010d} 00000 n \n".encode() for off in offsets)
        out.write(
            f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
    rgb_path.unlink()
    alpha_path.unlink()
