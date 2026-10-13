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
| `IMAGING_MAX_QA_JOBS` | `1` | `/qa/check` running at once per process, from its own slot so QA never takes a `/compose` slot. Over it: `429` + `Retry-After`. |
| `IMAGING_MAX_HEAVY_JOBS` | `2` | `/compose`, `/render/personalization`, `/photo/render`, `/photo/zip`, `/photo/scene-base`, and `/photo/scene-composite` running at once per process. Over it: `429` + `Retry-After`, before the body is read; the backend client retries. |
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
| `POST /qa/check` | `{file_key, target_width_in?, target_height_in?}` | `{width_px, height_px, has_alpha, soft_alpha_ratio, effective_dpi?, issues: [{code, severity, message, measured_in?, regions?}]}` |
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
- **Print-file checks (T-42-4, `app/qa_print.py`).** Four more issue codes, each at most once per
  call, with built-in limits only (no request fields). `measured_in` (float inches, unrounded) and
  `regions` (up to 20 boxes `{left, top, width, height}` as 0..1 ratios, merged, largest first)
  are left out when absent, never null.
  - `thin_line`: error under 0.02 in, warn from 0.02 up to but not including 0.03 in (so a 6 px
    stroke at 300 DPI is a warning and 9 px is clean). `measured_in` is the thinnest stroke.
    Method: the visible art (alpha >= 128; 30% after a shrink so a sub-pixel line is not lost) at
    an analysis resolution of at most 300 DPI and 16M pixels (a 22 x 60 in design is analysed at
    ~110 DPI, 0.009 in per pixel). A pixel's *chord* is its shortest run of art pixels in four
    directions (diagonals scaled by sqrt 2); pixels under 0.03 in form 8-connected components
    (touching pixels only, never across background, so halftone dots stay apart), and a component
    counts only with a bounding-box diagonal of at least **0.10 in** (`THIN_RUN_MIN_IN`), so the
    tapered tip of a thick shape (star point, script letter end, badge ring edge) is not a thin
    line. Its width is the component's median chord: exact on an axis-aligned stroke at native
    resolution, within one analysis pixel after a shrink, about 8% high at worst for an angled
    one. Known gap: a hairline shorter than 0.10 in is ignored; a point sharper than ~17 degrees
    is flagged. Needs a target size (width or height); without one there is no `thin_line` or
    `small_text`. Inches per pixel use `effective_dpi`, the same limiting axis as `low_dpi`.
  - `small_text`: warn only. Four or more shapes of similar height (0.4 to 2.5 x the row median),
    each 0.02 to under 0.10 in tall and at most 0.30 in wide, on one line with gaps up to 2.5 x
    their height. Letters that touch are one shape and are not seen (heuristic). Skipped with no
    issue when the art has over 1,000,000 regions or over 5,000 glyph-size shapes (a texture or
    halftone, not text; `MAX_TEXT_LABELS`, `MAX_TEXT_SHAPES`), so the work stays bounded.
  - `white_box`: warn when the file has real transparency and at least 95% of the outer ring (the
    larger of 2 px and 1% of the shorter side, on the analysis image) is opaque with R, G and B all
    245 or more. Four regions, one per side. An opaque file is `no_alpha` only.
  - `non_srgb`: warn for CMYK mode, a CMYK profile, or an RGB profile whose colorants differ from
    sRGB by more than 0.02 (Adobe RGB, Display P3). No profile, an sRGB profile, a gray profile
    or an RGB profile without readable colorants: no issue. Read from the file as loaded, before
    any conversion. No regions.
  - A fully transparent or fully opaque file skips the art checks. No network, no AI, no new
    dependency (numpy and pyvips only).
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
  OpenAI's convention: transparent = editable, opaque = keep; the opaque part is every pixel of
  the garment and the figure (any template coverage > 0, plus a disc rim of
  `MASK_SILHOUETTE_MARGIN_IN`, 0.0 in by measurement) and the print area plus 0.5 in (`protected_box_px`), so only the background is editable
  (T-38-1; GPT Image treats the mask as guidance only, so the two checks stay the authority). The base PNG carries its `garment/view` in a tEXt chunk, so the
  composite needs no garment/view. `/photo/scene-composite` treats the scene as untrusted:
  storage size ≤ 25 MB checked before download, magic bytes PNG/JPEG/WebP only, 256-4096 px a
  side and the decode pixel cap, else **400** `{detail}`. `print_box_px` must match the base's
  print area (±1 px) and `amazon_main` is refused (422). Then: (1) registration: the scene is
  cover-fitted to the base, then scale 0.90-1.10 and shift ≤ 8% are searched coarse to fine by
  masked normalized cross-correlation of edge maps over the garment and figure; (2) check 1
  `region_changed`: the scene's edges must sit on the base's garment outline (0.35 in band) in a
  6.5 in ring around the print box, score relative to the base itself, minimum
  **0.75** (and the search must not hit its limit); on failure `key` is null and nothing is
  written; (2b) also `region_changed`: nothing may be drawn into the print area (the provider
  ignored the mask but kept the outline): the share of the box (at 256 px) where the registered
  scene has a log-luminance edge ≥ 0.10/px that the blank base doesn't must be ≤ **0.004**.
  Relit, tinted, noisy, JPEG and ±30% folded scenes score 0.000 on navy, white, black and red;
  a disc with text bars 0.068-0.092, a small three-line wordmark 0.012-0.027; (3) the scene is
  warped (registration and upscale in one `lbb` resample: locally
  bounded bicubic, no halos) straight to the preset size and cropped; (4) the blank print area
  plus 0.5 in is redrawn from the template at output size, lit by the scene (global gain clamped
  0.5-1.6 with ≤ 15% per-channel tint, local shade low-frequency only: blurred 0.4 in and
  clamped 0.7-1.2, so scene detail never reaches the print), and the design printed at real
  size, displaced by the template folds plus the scene's light read as relief, wave 26
  underbase rule; (5) check 2 `design_drift`: the printed region is de-shaded and compared with
  the source design flat on the blank by SSIM (R, G, B mean, coverage-weighted, at 160 px),
  minimum **0.80**; on failure nothing is written. Thresholds were set from 32 test scenes (tee
  and hoodie, both views, navy and white): check 1 passes 0.92-1.00 for unchanged, mock-like
  perturbed (new background, relit, tinted, noise, 2% scale, shift, other provider size, JPEG)
  and strongly folded scenes; a 10% wider garment scores 0.01-0.15, a 15% shifted frame ≤ 0.33.
  Check 2: 0.94-1.00 unchanged or perturbed, 0.88-0.90 with ±30% folds over the print; a
  swapped design 0.36-0.42, the main color changed 0.69-0.71 (round 1 numbers), a disc and
  bars drawn over the print 0.60-0.70. Preset checks are reported in
  `preset_failures` (`background_not_white` on white-required presets, `too_small`,
  `design_larger_than_print_area`) and don't block the write.
- **Scene lock diagnostics and `structural` (T-37-1, ADR 0036).** The `checks` object of
  `/photo/scene-composite` also carries `region_reasons` (every failed sub-check of check 1, from
  `outline_low`, `registration_at_limit`, `box_foreign`, and since T-40-2 `placement_off`; empty when it passes), `box_foreign`
  (the 1b fraction) and `structural`; `registration` stays top level. `structural` is true when
  check 1 or 1b failed **and** registration's best edge correlation is below **0.5**
  (`STRUCTURAL_REG_MAX`): no scale + shift of the frame (searched 0.90-1.10, 8%) matches the
  base's garment, so the provider re-drew or re-posed it and a regenerate with the same request
  can't pass. It is decided from that one call's numbers only (same frame, same value, any
  preset), and false when the checks pass, for `design_drift` alone, for a garment present but
  displaced or widened (registration 0.71-0.81 in tests), and for art drawn into the box on an
  aligned garment (0.93). Calibration on the owner's 12 real `gpt-image-2` frames (white tee,
  flat_lay/home/studio; replay with `uv run python scripts/scene_calibrate.py <folder>`): all
  12 fail check 1 with registration 0.12-0.28, check-1 score <= 0.51 (mostly <= 0), box_foreign
  <= 0.0006; even a 6x wider search (scale 0.5-2.0, shift 30%, garment-only weights) finds no
  alignment above 0.49. These are real garment changes, not a strict check: **no threshold was
  changed**. All 12 are structural. Re-run the replay on new frames before changing the rule.
  Offline obedience simulation (T-38-1, `scripts/scene_mask_sim.py`; a provider that keeps every
  opaque mask pixel and draws a busy scene elsewhere, 144 cases): passes both checks everywhere in
  a darker room, and fails on a widened garment, a re-drawn garment and art in the print box. In a
  mid-tone room check 1 itself rejects a white garment on the white studio view even for a
  pixel-exact soft composite (0.58-0.75 vs 0.75), a background-contrast sensitivity of the check,
  not of the mask. The mask's hard edge costs little of that: an obedient frame scores at most
  **0.054** below the exact-composite oracle in the same room (0.036 on the white tee, white
  studio view; often above it), so near the line it can tip a case the oracle barely passes.
- **Studio scenes and the outline check (T-39-1, offline; check 1 not changed).** Wave 38's paid
  trial failed `studio` scenes in 3 of 4 (white tee 2 of 2). `scripts/scene_outline_diag.py
  <trial folders>` re-scores any frames and says why (garment fidelity after registration and
  the light gain: mean abs luma difference and SSIM over the kept garment with a luma affine fit
  removed; the figure separately; outline edge strength; share of the 0.35 in outer band that
  looks like the blank; the backdrop's luma). On the 18 real frames the kept garment matches the
  base in every failing studio frame (mean abs diff 0.7-3.5 levels, SSIM >= 0.96); the script's
  verdict, though, depends on three spill/backdrop thresholds that were moved after seeing the real
  frames (SPILL_MAX_DARK 0.03 -> 0.50, BACKDROP_APART 0.08 -> 0.15, redrawn only at spill >= 0.25;
  all toward "undecidable"). On the 6 failing studio frames the original values call 3 redrawn
  (both black-tee frames and white d6ea2019#2), 1 ours-low-contrast, 2 undecidable; the current
  ones call 0 redrawn, 1 ours-low-contrast, 5 undecidable. So no studio frame is called redrawn
  after the post-hoc change, and garment fidelity is good in 4 of 6 (8153e119#2 and d6ea2019#1
  sit in the undecidable band, mad 3.3-3.5); the failures are consistent with the outline: on the white tee the light gray backdrop
  (luma 0.76-0.82) sits within 7-25 levels of the drawn outline and its gradient falls to
  4.2-7.2 levels/px (passing frames 14.6-27.7, the base 12.8), and on the black tee the edge is
  there (63-80 vs 90) but displaced 2-4 px or haloed. Only flat_lay 43f8e7f1#1 (registration 0.42,
  at the limit) is a re-posed garment. `scene_mask_sim.py --sweep --bad` (1,716 frames: tee and
  hoodie, `lifestyle_base`, 11 seamless flat grays x 11 blanks, obedient and re-rendered
  providers, plus the known-bad frames) reproduces the dip with no redraw: a white tee fails
  only for a backdrop between sRGB levels 176 and 200 (luma 0.69-0.78; levels 206 and up, and
  170 and down, pass); the prompt's "light gray" as drawn is about 200. **Backdrop table**
  (check 1 on a flat gray, worst case over garments, blanks of the band, obedient and
  re-rendered; luma = level / 255 on `photos.luma`'s scale):

  | blank luma | tested blanks (luma) | best tone (worst-case c1) | every case >= 0.85 | fails (< 0.75) |
  | --- | --- | --- | --- | --- |
  | white [0.85, 1.0] | 0.949 | 250 / 0.98 (1.000; 235 and 215 tie) | 215-250 and 10-160 | 190-200 (min 0.389) |
  | light [0.55, 0.85) | 0.792, 0.722, 0.624 | 215 / 0.84 (0.989) | 190-250 and 10-64 | 128-160 (min 0.466) |
  | mid [0.22, 0.55) | 0.502, 0.370, 0.292, 0.264 | 215 / 0.84 (0.993) | 160-250 | 32-96 (min 0.613) |
  | dark [0.10, 0.22) | 0.209, 0.163 | 215 / 0.84 (0.994) | 128-250 | 10-64 (min 0.608) |
  | black [0, 0.10) | 0.067 | 215 / 0.84 (0.995) | 96-250 | 10-32 (min 0.614) |

  A single backdrop of level 235 (luma 0.92) is inside every band's every-case window. Known-bad
  frames (widened or narrowed 10%, another garment, garment absent, widened on a backdrop of the
  blank's own luma) score at most **0.689** on any backdrop (narrowed 10%, sport gray; the
  hoodie moved 0.6 in down 0.67), so today's gap to 0.75 is **0.061**, not the 0.42 measured in
  wave 27; the sim's `b-shift` (moved 0.6 in on an unmoved figure) passes check 1 in 98 of 242
  rows and is not asserted. Real redraws: the 12 wave 37 frames score at most 0.513 and the 6
  wave 37 trial A frames 0.128. A measure that subtracts the scene's background noise floor was
  tried offline and lowers the unlucky-backdrop scores instead of raising them (0.648 -> 0.641,
  0.606, 0.552 at 1, 2 and 3 times the floor), and one that normalizes by the base's outline contrast has nothing to
  normalize on the white tee (its own outline step is 11.7 levels): **the lever is the backdrop
  tone in the prompt, not the check, and no threshold or measure was changed.** The default
  `scene_mask_sim.py` run now exits 0 when every row is expected or within the mask-edge cost of
  the oracle (156 cases with `--frames`).
- **Check 1 margin and the placement guard, check 1c (T-40-2, ADR 0039).** Measured offline with
  `scripts/scene_guard_sweep.py` (sets: 8 real wave 39 passes of 12 frames; known-good = obedient,
  re-rendered, mock-like, whole-frame shift and 0.92-1.08 scale, provider sizes 1024² and
  1536×1024, folds; known-bad = garment narrowed 5/10/15%, widened 10%, moved 0.3/0.6/1.0 in on
  each axis both ways, scaled 0.9/1.1; tee and hoodie, six blanks, backdrops 105/200/235; 1,296
  known-bad and 528 known-good frames over `lifestyle_base` and `on_model_white`). **The margin
  the old comment gave (0.42) was wrong**: lowest real pass 0.822, lowest synthetic pass 0.760;
  highest narrowed or widened garment **0.689** (narrowed 10%, sport-gray tee, backdrop 235), gap
  to 0.75 = **0.061** (8% of the threshold; 0.072 below the lowest real pass). Row counts, check
  1 alone: 0 of 288 reshaped rows (narrowed, widened) pass; garment-only *moves and scales* pass
  in 305 of 1,008 (they score up to 1.000): registration absorbs a rigid shift and 0.90-1.10
  scale on purpose, on `lifestyle_base` the garment is the whole product, and the print follows
  it, so the output is right (hence not harmful, not detectable by any outline measure after
  registration). Check 1c measures what registration cannot absorb: after registration, the
  garment's outline pixels (coverage-mask boundary in the check-1 ring) are split by the side
  they face; for each side the scene's edge strength is sampled at the outline displaced up to
  2 in along the axis and the best displacement is that side's offset. Width = right - left, height
  = bottom - top (positive = bigger), shift = their means, in inches at the template ppi; a side
  with no edge (best mean edge < 0.2) counts 0. `placement_off` is added when
  max(|width|, |height|) > **0.5 in** (`PLACEMENT_SIZE_MAX_IN`) or max(|x|, |y|) > **0.15 in**
  (`PLACEMENT_SHIFT_MAX_IN`). Margins (A6, in inches and as a share of the threshold):
  | measure | threshold | highest known-good | gap_good | lowest narrowed/widened 10-15% | gap_bad |
  | --- | --- | --- | --- | --- | --- |
  | size | 0.5 | 0.299 (real passes 0.132; frames that pass check 1 0.221) | 0.20 = 40% | 0.760 | 0.26 = 52% |
  | shift | 0.15 | 0.076 (real passes 0.058) | 0.074 = 49% | residual of partly absorbed moves 0.16-0.20 | 0.01-0.05 = 7-34% |
  Narrowed 5%, the smallest redraw, measures 0.36-1.17 in (`lifestyle_base`: 23 of 36 rows over the
  threshold, 20 with 25% room); every one of those rows already fails check 1 (c1 <= 0.62) and still does.
  Result: the guard changes no real-frame verdict (8 of 12 pass before and after; the two real
  frames it flags, 394d6dc5#2 and 3457ac2a#1, already failed), none of 528 known-good frames, and
  turns 4 of 305 passing garment-only moves (residual offset 0.16-0.20 in) into failures. **Not
  detectable, listed instead of tuned:** the other 301 rigid move and scale rows
  (`move-x/y +-0.3/0.6/1.0`, `gscale0.9/1.1`; 147 on `lifestyle_base`, 154 on `on_model_white`
  where the garment moves against a still figure and registration still takes it). Cost:
  `placement_check` is ~46 ms of a ~0.8 s composite at 1024²; peak RSS unchanged (about 540 MB in the
  test process, no new full-size array). `is_structural` is unchanged (`reasons` and registration
  < 0.5): a placement-only failure with a good registration is retried. Replay on new frames:
  `scripts/scene_guard_sweep.py --real <folders> --out f.tsv`, then `--summarize f.tsv ...`.

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

**T-42-4 (print-file checks in `/qa/check`):** a 22 x 60 in design at 300 DPI (6,600 x 18,000 px,
119M pixels) on the same laptop: 3.1 s and 667 MB peak RSS for a dense full-color file (459 MB
PNG), 0.7 s and 640 MB for a mostly transparent one; base was 1.3 s and 77 MB without these
checks (it only decodes strips). Budget for this check: peak RSS under 1.5 GB, under 20 s. The
extra memory is the analysis image (16M pixels x 4 bands) plus a few uint8 and uint32 planes of
the same size; it does not grow with the file, because the analysis size is capped. Round 2
(security S-72/S-73): run lengths in row blocks, per-component boxes and chord medians from
bounded per-block arrays (no int64 index array over all pixels), at most 300 thin components
measured, and `small_text` caps above. Re-measured: 22 x 60 in mostly transparent 0.5 s / 608 MB;
22 x 60 in 0.04 in dot grid 1.8 s / 814 MB (no `thin_line`); 22 x 60 in random noise (302 MB PNG)
2.9 s / 743 MB. Small hostile files at a 40 in target, each ~100 KB: 1 px checkerboard 0.7 s /
547 MB (was 10 s / 1.9 GB), 2 x 2 px dots on a 3 px pitch 1.5 s / 921 MB (was 39 s / 1.6 GB),
8,000,000 x 2 px strip 0.1 s / 930 MB (the base decode alone is ~670 MB). `/qa/check` has its own
limiter, so at most `IMAGING_MAX_QA_JOBS` (default 1) run at once per process and none takes a compose slot. A file over
`imaging_max_decode_pixels` still answers with the `unreadable` issue before any of this runs.

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
design 10×10 in): scene base 20-75 ms; composite 0.65-0.77 s per call (+~46 ms for check 1c in T-40-2: registration, check 1,
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
app/qa_print.py    thin line, small text, white box, color profile checks
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
