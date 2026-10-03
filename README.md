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
| `IMAGING_MAX_HEAVY_JOBS` | `2` | `/compose`, `/render/personalization`, `/photo/render`, `/photo/zip`, `/photo/scene-base` and `/photo/scene-composite` running at once per process. Over it: `429` + `Retry-After`, before the body is read; the backend client retries. |
| `IMAGING_RETRY_AFTER_S` | `5` | The `Retry-After` value. |
| `IMAGING_MAX_DECODE_PIXELS` | `570000000` | Pixel cap checked from the header before decoding (raster, SVG size, and PDF page size × DPI). |
| `IMAGING_PHOTO_TEMPLATE_CACHE` | `4` | Listing-photo templates kept drawn per (garment, view, size), about 20–40 MB each at 2000 px. |

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
| `POST /nest` | `{items: [{id, width_in, height_in, quantity=1}], sheet_width_in=22, spacing_in=0.25, margin_in=0.25, max_length_in=240, allow_rotation=true, label_height_in=0.42}` | `{sheets: [{index, length_in, utilization, placements: [{id, copy, x_in, y_in, width_in, height_in, rotated}]}], overall_utilization}` |
| `POST /compose` | `{width_in, length_in, dpi=300, placements: [{transfer_id, file_key, x_in, y_in, width_in, height_in, rotated, label: {order_no, item_no, size, color, design, reprint}}], label_height_in=0.42, out_key, pdf_key?, preview_key, preview_width_px=1200, mirror=false}` | `{key, pdf_key?, preview_key, width_px, height_px, bytes}` |
| `POST /preview` | `{file_key, out_key, max_px=512 (64..2048)}` | `{out_key, width_px, height_px}` |
| `POST /mockup` | `{design_key, blank_color_hex, placement: "front"\|"back", out_key, size_px=1200}` | `{key}` |
| `POST /photo/palette` | `{design_key}` | `{colors: [{hex, share}], light_share, dark_share, transparent_share}` |
| `GET /photo/templates` | | `[{garment, view, placement, print_areas: {front\|back: {w_in, h_in}}, drawn}]` |
| `POST /photo/render` | `{design_key, design_width_in, design_height_in, placement, garment, view, blank_hex, underbase_preview=false, preset, out_key, xmp_subjects=[]}` | `{key, width_px, height_px, format, print_box_px: [x,y,w,h], checks: {passes, failures, background_pure_white, fill_ratio, longest_side_px}, print_area_px, scale, underbase_applied, bytes, render_ms}` |
| `POST /photo/zip` | `{items: [{key, name}], out_key}` | `{key, bytes}` |
| `POST /photo/scene-base` | `{garment, view: on_model_white\|lifestyle_base, blank_hex, size_px=1024 (or [w, h], 256-2048), out_key (.png), mask_out_key (.png)}` | `{key, mask_key, print_box_px: [x,y,w,h], width_px, height_px, protected_box_px, ppi}` |
| `POST /photo/scene-composite` | `{scene_key, base_key, print_box_px, design_key, design_width_in, design_height_in, placement=front, blank_hex, preset, out_key, xmp_subjects=[], garment?, view?, underbase_preview=false}` | `{key (null when a check fails), checks: {passes, failures, region_unchanged_score, design_lock_score, region_unchanged_min, design_lock_min}, width_px, height_px, format, print_box_px, preset_failures, registration, upscale, upscale_method, underbase_applied, bytes, render_ms}` |
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
  `overall_utilization` (B-41) is the length-weighted film use across every sheet in the set --
  total design area over total film used, not a plain average of each sheet's own
  `utilization` -- so a lightly-filled last sheet pulls it down by how much film it actually took,
  not by counting as much as a full one. Additive alongside `sheets`.
- **Compose.** Pass the nest placements through unchanged (plus `transfer_id`, `file_key`,
  `label`). The art is fitted inside the placement box (aspect kept, centered) and rotated 90°
  clockwise when `rotated`. Under each design, inside the label strip: an opaque white QR code
  encoding `transfer_id` exactly, then `order_no · item_no · size · color · design` (shrunk then
  ellipsized to the design's width) and a red `REPRINT` line when flagged. `label_height_in`
  (default 0.42) must match what was passed to `/nest`. The PNG is RGBA with a transparent
  background and 300 DPI metadata; the PDF is one page at the exact physical size (sheets over
  200in use PDF `/UserUnit`, since plain PDF pages top out at 200in); the preview is
  `preview_width_px` wide, transparent, with the sheet's aspect ratio. `mirror` (default `false`)
  flips only the design artwork horizontally, for DTF vendors whose press/RIP expects a
  pre-mirrored print -- it never touches the label strip, header or cut guides, so a presser can
  still read and scan them on the sheet. Any input carrying an embedded ICC profile (including a
  profiled CMYK file) is color-managed through it; CMYK without a profile still converts, never
  rejected, through libvips' built-in formula (checked live against its own no-profile fallback:
  same numbers). The profile check itself is cheap, but the real color transform is skipped
  whenever a file has no profile (the common case) -- paying it unconditionally cost measured
  ~2ms on a 1.3MP design and ~20ms on a 15MP one, a budget-breaking slowdown across a full sheet.
- **Preview.** A small sRGB PNG thumbnail of any supported input (design art, a PDF's first
  page, ...), for catalog/order-item previews (T-P1-4). Longest side is at most `max_px`, never
  upscaled; vector input (PDF/SVG) rasterizes at QA's own recommended print DPI (300), same as
  `/qa/check`'s own default.
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
  drawn), `suspicious_chars` (emoji, symbols, non-Latin scripts, control characters). A photo
  slot also flags `aspect_mismatch` (the photo's aspect ratio is far enough from the slot's that
  it will be heavily letterboxed or cropped) and `upscale` (the photo must be stretched more than
  1.5x to fill the slot at the render DPI, on top of the existing `low_res_photo` for an
  absolute DPI floor). A `background_key` image is stretched to the template size.
- **Listing photos (T-26-2).** `/photo/*` keys must be `<companyId uuid>/...`; every input key
  must share `out_key`'s company prefix, else **400** `{detail}` (the one non-422 error here).
  Templates: `tee`, `hoodie`, `crewneck`, `tank` × `front_flat`, `folded`, `back`,
  `on_model_white`, all drawn in code (`app/garments.py`, inches on an adult L; the on-model
  figure is a neutral grey illustration, not photoreal). Print areas: tee front 12×16, back 14×16;
  hoodie front 12×9 (above the pocket), back 14×16; crewneck front 12×14, back 14×16; tank front
  10×13, back 10×14. `view: back` needs `placement: back` (the back print file; without one the
  backend skips that view), every other view `front` (422 otherwise). The design covers exactly
  `design_width_in × design_height_in` at the template's scale, horizontally centered on the
  print area and top-aligned to it; it is never scaled up, and a design larger than the print
  area is scaled down (`scale` < 1) with `design_larger_than_print_area` in `failures`. On
  `folded` the part below the fold is hidden, as on a real folded shirt. The blank is exactly
  `blank_hex` wherever the template is unshaded; folds bend (displace) and darken the print.
  `underbase_preview` on a blank with sRGB luma < 0.45 shows semi-transparent design pixels over
  white ink (alpha ≥ 25% gets full underbase); on lighter blanks it changes nothing.
  Presets: `amazon_main` 2000×2000 (white required, fill ≥ 0.85, longest ≥ 1600, photo
  required), `amazon_alt` 2000×2000 (≥ 1000), `etsy` 2700×2025 (≥ 2000), `shopify` 2048×2048,
  `tiktok` 1600×1600 (≥ 600), `walmart` 2000×2000 (white required, ≥ 1500). Output is JPEG (Q92,
  4:4:4, sRGB ICC embedded), or PNG when `out_key` ends in `.png`, with the product on a pure RGB
  255 background. Checks run on the encoded file: `background_pure_white` = every pixel outside
  the product (grown to whole 8×8 JPEG blocks, where compression ringing lives) is exactly
  255,255,255; `fill_ratio` = the product bounding box's share of the frame on its limiting side
  (the product spans 90%; on-model 95%, where the legs leave the frame); `longest_side_px`.
  Failure codes: `design_larger_than_print_area`, `illustration_not_photo` (drawn templates,
  `amazon_main` only), `background_not_white`, `fill_below_min`, `too_small`; `passes` = no
  failures. `xmp_subjects` (≤ 20, 1–100 chars) are written as XMP `dc:subject` (JPEG APP1, PNG
  iTXt `XML:com.adobe.xmp`); an empty list writes no XMP. Palette: up to 6 alpha-weighted colors
  whose shares sum to 1 over opaque pixels (the tail is folded into the nearest kept color);
  light = luma ≥ 0.75, dark ≤ 0.25; deterministic. Zip: stdlib, names sanitised (no `..`, no
  absolute paths, odd characters → `_`, duplicates get `-2`), images stored, others deflated.
- **Scene lock (T-27-2, ADR 0023 phase B).** `/photo/scene-base` draws the blank garment
  (no design) for the image provider: `on_model_white`, or `lifestyle_base` (the same figure at
  62% of the frame on a neutral 228,226,222 set, leaving room for a scene). The mask follows
  OpenAI's convention: transparent = editable, opaque = keep; only the print area plus 0.5 in
  (`protected_box_px`) is opaque. The base PNG carries its `garment/view` in a tEXt chunk, so the
  composite needs no garment/view. `/photo/scene-composite` treats the scene as untrusted:
  storage size ≤ 25 MB checked before download, magic bytes PNG/JPEG/WebP only, 256-4096 px a
  side and the decode pixel cap, else **400** `{detail}`. `print_box_px` must match the base's
  print area (±1 px) and `amazon_main` is refused (422). Then: (1) registration: the scene is
  cover-fitted to the base, then scale 0.90-1.10 and shift ≤ 8% are searched coarse to fine by
  masked normalized cross-correlation of edge maps over the garment and figure; (2) check 1
  `region_changed`: the scene's edges must sit on the base's garment outline (0.35 in band) in a
  6.5 in ring around the print box, score relative to the base itself, minimum
  **0.75** (and the search must not hit its limit); on failure `key` is null and nothing is
  written; (3) the scene is warped (registration and upscale in one `lbb` resample: locally
  bounded bicubic, no halos) straight to the preset size and cropped; (4) the blank print area
  plus 0.5 in is redrawn from the template at output size, lit by the scene (global gain clamped
  0.5-1.6 with ≤ 15% per-channel tint, local shade 0.6-1.25), and the design printed at real
  size, displaced by the template folds plus the scene's light read as relief, wave 26
  underbase rule; (5) check 2 `design_drift`: the printed region is de-shaded and compared with
  the source design flat on the blank by SSIM (R, G, B mean, coverage-weighted, at 160 px),
  minimum **0.80**; on failure nothing is written. Thresholds were set from 32 test scenes (tee
  and hoodie, both views, navy and white): check 1 passes 0.92-1.00 for unchanged, mock-like
  perturbed (new background, relit, tinted, noise, 2% scale, shift, other provider size, JPEG)
  and strongly folded scenes; a 10% wider garment scores 0.01-0.15, a 15% shifted frame ≤ 0.33.
  Check 2: 0.94-1.00 unchanged or perturbed, 0.88-0.90 with ±30% folds over the print; a
  swapped design 0.36-0.42, the main color changed 0.69-0.71. Preset checks are reported in
  `preset_failures` (`background_not_white` on white-required presets, `too_small`,
  `design_larger_than_print_area`) and don't block the write.
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

**T-P1-2 (ICC-aware `to_srgba`, mirror, `/preview`, `overall_utilization`):** re-measured on the
same laptop with a realistic 93-design, 22in-wide sheet set (same size mix as
`tests/test_nesting.py`'s `test_realistic_size_mix_nests_dense`, split by `/nest` into sheets of
236-239in and one 56in sheet): 40 designs at 238.9in, PNG + preview, 397 MB peak RSS; 15 designs
at 56.5in, PNG + PDF + preview, 477 MB. Both within the existing budget (no RSS regression). Wall
time on this run (5.5-6.5s for the long sheet) was noisy on a shared dev machine, so it isn't a
clean before/after; the actual cost driver was isolated instead: `to_srgba` now calls
`icc_transform` only when a file carries an embedded ICC profile (`get_typeof("icc-profile-data")
!= 0`), confirmed by microbenchmark to cost ~0.03ms/call for a design with no profile (same as
before this change) rather than ~2-20ms/call if paid unconditionally -- the first version of this
change called `icc_transform` on every image and was reverted for exactly that reason. `/preview`
on a single 1200x900 design: ~27ms, no measurable RSS impact (it never loads a full sheet).

**T-26-2 listing photos** (same laptop, `uv run python bench/photo_bench.py`, sample art
3000×3600 px placed at 10×12 in, `amazon_alt` 2000×2000 JPEG): render 0.18–0.23 s average with
the template drawn (cold, max 0.38 s), 0.09–0.11 s with it cached. The first render in a fresh
process peaks at ~305 MB RSS (from a 58 MB base: +80 template draw, +110 design decode and vips
thread start, +55 encode). Across all 16 templates RSS climbs to a 570–700 MB high-water mark
(4 cached templates plus allocator high-water; a cache of 1 still plateaus near 455 MB), so a
photo-set job should count on ~0.7 GB, like a compose. Zip of 48 renders (8.8 MB): 0.01 s.

**T-27-2 scene composite** (same laptop, live on `:8141` with MinIO, `shopify` 2048×2048 JPEG,
design 10×10 in): scene base 20-75 ms; composite 0.65-0.77 s per call (registration, check 1,
warp to 2048, restore and print, encode, check 2), across tee `on_model_white` and hoodie
`lifestyle_base`, light and dark blanks, provider sizes 1024², 1024×1536 and 1536×1024. Peak RSS
of the server process over those 4 bases + 4 composites + 1 refused: 637 MB (template cache of 4
included), so count on ~0.7 GB per process, like a photo-set job.

## Layout

```
app/main.py        routes, request models, 422 mapping
app/nesting.py     MaxRects nesting with label strips, rotation, multi-sheet
app/compose.py     streaming gang-sheet compose, QR + text labels
app/pdf.py         streaming RGBA -> PDF writer (true physical size)
app/qa.py          DPI / alpha checks, clean-alpha
app/render.py      personalization templates
app/mockups.py     vector t-shirt mockups
app/garments.py    listing-photo garment templates (4 garments x 4 views, inches)
app/photos.py      listing photos: composite, presets and checks, XMP, palette, zip
app/keys.py        company-prefix checks for /photo/* keys
app/scenes.py      AI lifestyle scenes: base + edit mask, registration, two lock checks
bench/             photo_bench.py: listing-photo budgets and contact sheet
app/labels.py      4x6 mock shipping label (reportlab)
app/sample_art.py  seed artwork
app/fonts.py       bundled OFL fonts (app/fonts, licenses alongside)
app/storage.py     S3 / local storage backends
```

Text is drawn with Pillow/FreeType. The pyvips wheel can render text with a `fontfile`, but it
goes through Pango + fontconfig (which warns about missing config in slim images) and has no
stroke support.
