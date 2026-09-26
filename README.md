# invai-imaging

Python 3.13 + FastAPI + pyvips service for everything image-heavy: gang-sheet nesting and
composing, print-file QA, personalization rendering, mockups, sample art and mock shipping labels.
It implements `invai-docs/build/v1-plan.md` section 5.1. The backend calls it over HTTP; every
input and output is a storage key (S3/MinIO), never an image in the request body.

## Run

```
uv sync --all-groups
cp .env.example .env            # defaults already point at the local MinIO
uv run uvicorn app.main:app --port 8000
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Needs the local MinIO from `invai-infra/local` (`docker compose up -d`). The bucket (`invai-local`)
is created at startup if missing. `STORAGE=local` swaps S3 for a directory (`LOCAL_STORAGE_DIR`),
which the tests use. pyvips ships libvips in its binary wheel, so no system packages are needed
(also in the Docker image).

## Endpoints

Units are inches unless named `_px`. Every error is `422 {"detail": "<message>"}`, including
request validation errors and missing keys (`file not found: <key>`).

### Auth, limits and config (T-9-5)

| Env | Default | What |
| --- | --- | --- |
| `IMAGING_SHARED_SECRET` | `invai-imaging-dev-secret` | Every path but `/health` needs `X-Imaging-Secret: <secret>` (constant-time compare), else `401`. A comma-separated list accepts each entry (rotation). The backend sends its own `IMAGING_SHARED_SECRET`. |
| `IMAGING_ENV` | `development` (`production` in the Docker image) | `production` refuses to start on the dev secret or one under 32 chars. |
| `IMAGING_DEV_ENDPOINTS` | `false` | `/labels/mock` and `/sample-art` answer `404` unless `true`. Local dev (`.env.example`) sets it; the backend's mock carrier and the seed need it. |
| `IMAGING_MAX_HEAVY_JOBS` | `2` | `/compose` and `/render/personalization` running at once per process. Over it: `429` + `Retry-After`, before the body is read; the backend client retries. |
| `IMAGING_RETRY_AFTER_S` | `5` | The `Retry-After` value. |
| `IMAGING_MAX_DECODE_PIXELS` | `570000000` | Pixel cap checked from the header before decoding (raster, SVG size, and PDF page size × DPI). |

- Inputs are PNG, JPEG, WebP, TIFF, SVG or PDF by magic bytes, loaded with that format's loader
  only (libvips' untrusted loaders are blocked). A key with an extension (`…/<uuid>.png`) must
  match its bytes. QA reports a mismatch as an `unreadable` issue; other endpoints answer `422`.
- Bounds: sheet width ≤ 60 in, length ≤ 240 in, DPI 36–1200, at most 10,000 placements per
  compose or copies per nest, keys ≤ 512 chars.
- Every response has `X-Render-Ms`; `/compose` and `/render/personalization` also return
  `render_ms`. The Docker image has a `HEALTHCHECK` on `/health` and waits up to 30 s for running
  requests on shutdown.

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | | `{ok, vips_version}` |
| `POST /qa/check` | `{file_key, target_width_in?, target_height_in?}` | `{width_px, height_px, has_alpha, soft_alpha_ratio, effective_dpi?, issues: [{code, severity, message}]}` |
| `POST /qa/clean-alpha` | `{file_key, out_key, threshold=128}` | `{key}` |
| `POST /render/personalization` | `{template: {width_in, height_in, background_key?, slots: [...]}, values, out_key, dpi=300}` | `{key, width_px, height_px, flags: [{slot, code, message}]}` |
| `POST /nest` | `{items: [{id, width_in, height_in, quantity=1}], sheet_width_in=22, spacing_in=0.25, margin_in=0.25, max_length_in=240, allow_rotation=true, label_height_in=0.42}` | `{sheets: [{index, length_in, utilization, placements: [{id, copy, x_in, y_in, width_in, height_in, rotated}]}]}` |
| `POST /compose` | `{width_in, length_in, dpi=300, placements: [{transfer_id, file_key, x_in, y_in, width_in, height_in, rotated, label: {order_no, item_no, size, color, design, reprint}}], label_height_in=0.42, out_key, pdf_key?, preview_key, preview_width_px=1200}` | `{key, pdf_key?, preview_key, width_px, height_px, bytes}` |
| `POST /mockup` | `{design_key, blank_color_hex, placement: "front"\|"back", out_key, size_px=1200}` | `{key}` |
| `POST /labels/mock` | `{shipment_id, carrier, service, tracking_code, from, to, weight_oz, out_key}` (addresses: `name, city, state, zip`) | `{key}` |
| `POST /sample-art` | `{text, out_key, width_in, height_in, dpi=300, color_hex="#111111"}` | `{key}` |

### Details the caller should know

- **Nesting.** Each design's footprint is `(w + spacing) x (h + label_height + spacing)`: the order
  label strip always sits under the design as placed, so a rotated design occupies
  `(h, w + label)`. Margins stay clear on all four sides. `placements[].width_in/height_in` are the
  placed (post-rotation) size; `x_in/y_in` are the design's top-left from the sheet's top-left.
  `copy` is 0-based (0..quantity-1). `length_in` is the true used length (last label + bottom
  margin), not `max_length_in`; `utilization` = sum of design areas / (sheet width x length).
  Items that don't fit go to the next sheet; an item larger than the sheet in both orientations
  is a 422. Several MaxRects heuristics and orderings run per sheet and the best one wins.
- **Compose.** Pass the nest placements through unchanged (plus `transfer_id`, `file_key`,
  `label`). The art is fitted inside the placement box (aspect kept, centered) and rotated 90°
  clockwise when `rotated`. Under each design, inside the label strip: an opaque white QR code
  encoding `transfer_id` exactly, then `order_no · item_no · size · color · design` (shrunk then
  ellipsized to the design's width) and a red `REPRINT` line when flagged. `label_height_in`
  (default 0.42) must match what was passed to `/nest`. The PNG is RGBA with a transparent
  background and 300 DPI metadata; the PDF is one page at the exact physical size (sheets over
  200in use PDF `/UserUnit`, since plain PDF pages top out at 200in); the preview is
  `preview_width_px` wide, transparent, with the sheet's aspect ratio.
- **QA.** `effective_dpi` = pixels / target inches on the limiting axis. `low_dpi` is an error
  under 150 and a warning under 300. `soft_alpha_ratio` = pixels with 0 < alpha < 255 / pixels with
  alpha > 0; `soft_alpha` warns above 5% (anti-aliased edges alone are usually 1–3%). `no_alpha`
  warns when there is no alpha channel or it is fully opaque. `tiny_file` warns under 150px or
  2 KB. `unreadable` is returned as an issue (200), not a 422, when the key exists but isn't an
  image. `clean-alpha` snaps alpha to 0/255 at `threshold`.
- **Personalization.** Fonts: `Inter`, `Inter Bold`, `Inter Black`, `Oswald`, `Pacifico`,
  `Bebas Neue` (unknown names fall back to Inter). Text is centered vertically in its slot and
  aligned `left|center|right`. If it doesn't fit it shrinks in 2% steps down to 60% of
  `font_size_pt`; past that it renders at 60%, is clipped to the slot and gets `overflow`. Other
  flags: `empty` (missing/blank value; nothing drawn), `too_long` (over `max_chars`; still
  drawn), `suspicious_chars` (emoji, symbols, non-Latin scripts, control characters). A
  `background_key` image is stretched to the template size.
- **Mockup.** A vector-drawn tee (front or back neckline), tinted to `blank_color_hex` with fold
  shading, the design fitted into the chest box. Opaque PNG on a light gray background.
- **Mock label.** 4x6in PDF: service box, carrier/service, weight, from/to, Code128 tracking
  barcode, and a "SAMPLE — NOT A REAL LABEL" watermark.
- **Sample art.** Deterministic per text: one of four styles (stacked slab, badge frame, stars
  rule, script) in the bundled fonts with a hard shadow and outline, transparent background.

## Performance

A full sheet, measured on an Apple Silicon laptop (93 designs from 20 sample-art
files, 22 x 239.85in at 300 DPI = 6,600 x 71,955 px):

| Output | Time | Peak RSS |
| --- | --- | --- |
| PNG + preview | 4.1 s | 440 MB |
| PNG + PDF + preview | 10.2 s | 495 MB |

The process idles at about 75 MB. Memory is flat in sheet length because the sheet is never held
in memory: designs are pre-scaled into temp PNGs, the sheet is a lazy pyvips `black` +
`insert` graph streamed to `pngsave` top to bottom, and the PDF is streamed from that PNG in strips.
Temp disk use is about the output size plus the pre-scaled designs. Dense, full-color artwork
compresses worse than the sample art, so expect larger files and somewhat longer PNG encodes.

## Layout

```
app/main.py        routes, request models, 422 mapping
app/nesting.py     MaxRects nesting with label strips, rotation, multi-sheet
app/compose.py     streaming gang-sheet compose, QR + text labels
app/pdf.py         streaming RGBA -> PDF writer (true physical size)
app/qa.py          DPI / alpha checks, clean-alpha
app/render.py      personalization templates
app/mockups.py     vector t-shirt mockups
app/labels.py      4x6 mock shipping label (reportlab)
app/sample_art.py  seed artwork
app/fonts.py       bundled OFL fonts (app/fonts, licenses alongside)
app/storage.py     S3 / local storage backends
```

Text is drawn with Pillow/FreeType. The pyvips wheel can render text with a `fontfile`, but it
goes through Pango + fontconfig (which warns about missing config in slim images) and has no
stroke support.
