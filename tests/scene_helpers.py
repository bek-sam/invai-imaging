"""Synthetic provider scenes for the scene-lock tests (T-27-2): what an image model might send
back for a scene base, from untouched to a moved or reshaped garment."""

import dataclasses

import pyvips

from app import photos, scenes
from app.fonts import parse_hex
from app.garments import template_maps


def base_image(garment: str, view: str, w: int, h: int, blank: str) -> pyvips.Image:
    res = scenes.render_base(garment, view, w, h, blank)
    return pyvips.Image.new_from_buffer(res.png, "")[0:3]


def _background(w: int, h: int) -> pyvips.Image:
    """A 'room': a warm vertical gradient with a window-like block and some texture."""
    xyz = pyvips.Image.xyz(w, h)
    yy = xyz[1].cast("float") / h
    xx = xyz[0].cast("float") / w
    r = 150 + 70 * yy + 10 * (xx * 2100).sin()  # vips sin() is degrees
    g = 130 + 60 * yy + 8 * (yy * 3000).sin()
    b = 110 + 40 * yy
    bg = r.bandjoin([g, b])
    return bg.draw_rect([235, 238, 240], round(0.08 * w), round(0.1 * h), w // 5, h // 4, fill=True)


def perturbed(
    base: pyvips.Image,
    garment: str,
    view: str,
    out_size: tuple[int, int],
    scale: float = 1.02,
    shift: tuple[float, float] = (7, -5),
    gain: float = 0.92,
    tint=(1.0, 0.97, 0.92),
    noise: float = 3.0,
    new_background: bool = True,
) -> pyvips.Image:
    """The mock-like whole-frame perturbation: a new background where the base has none, relit
    and tinted, noise, a small scale and shift, then resized to a provider size."""
    w, h = base.width, base.height
    maps = template_maps(garment, view, w, h)
    img = base.cast("float")
    if new_background:
        img = maps.product.ifthenelse(img, _background(w, h), blend=True)
    img = img * [gain * t for t in tint]
    if noise:
        img = img + pyvips.Image.gaussnoise(w, h, sigma=noise, mean=0, seed=7)
    # Scale about the center, then shift.
    off = ((1 - scale) * w / 2 + shift[0], (1 - scale) * h / 2 + shift[1])
    img = scenes._warp(img, scale, off, w, h)
    # A provider size of another aspect: the frame is extended (outpainted) around the base.
    ow, oh = out_size
    fit = min(ow / w, oh / h)
    img = scenes._warp(img, fit, ((ow - w * fit) / 2, (oh - h * fit) / 2), ow, oh)
    img = (img > 255).ifthenelse(255, (img < 0).ifthenelse(0, img))
    return img.cast("uchar").copy(interpretation="srgb")


def moved_garment(
    garment: str,
    view: str,
    w: int,
    h: int,
    blank: str,
    dx_in: float = 0.0,
    dy_in: float = 0.0,
    widen: float = 1.0,
) -> pyvips.Image:
    """The garment moved (inches) or widened on an unmoved figure and background."""
    maps = template_maps(garment, view, w, h)
    rgb = tuple(parse_hex(blank)[:3])
    fabric = photos._garment_rgb(maps, rgb).rint().cast("uchar")

    def move(img: pyvips.Image, extend: str) -> pyvips.Image:
        if widen != 1.0:
            cx = maps.product_bbox_px[0] + maps.product_bbox_px[2] / 2
            img = scenes._warp(img, 1.0, (0, 0), w, h).affine(
                [widen, 0, 0, 1],
                oarea=[0, 0, w, h],
                odx=(1 - widen) * cx,
                extend=extend,
            )
        dx, dy = round(dx_in * maps.ppi), round(dy_in * maps.ppi)
        return img.embed(dx, dy, w, h, extend=extend) if dx or dy else img

    moved = dataclasses.replace(maps, front=move(maps.front, "black"))
    bg = scenes.LIFESTYLE_BG if view == "lifestyle_base" else (255, 255, 255)
    return photos.layer(moved, fabric, move(fabric, "copy"), bg)


def folds(img: pyvips.Image, box: tuple[int, int, int, int], depth: float = 0.15) -> pyvips.Image:
    """Diagonal soft fold shading (+-depth) over a region, like light on a worn garment."""
    w, h = img.width, img.height
    x, y, bw, bh = box
    xyz = pyvips.Image.xyz(w, h).cast("float")
    wave = ((xyz[0] + xyz[1] * 0.6) * (360 / max(20, bw / 3))).sin()  # vips sin() is degrees
    inside = pyvips.Image.black(w, h).draw_rect(255, x, y, bw, bh, fill=True).gaussblur(4) / 255
    factor = 1 + depth * wave * inside
    out = img.cast("float") * factor
    return (out > 255).ifthenelse(255, out).cast("uchar").copy(interpretation="srgb")


def drawn_marks(
    img: pyvips.Image, box: tuple[int, int, int, int], ink, kind: str = "logo"
) -> pyvips.Image:
    """The provider ignored the mask and drew into the print box, garment outline intact:
    `logo` is a disc and four text-like bars (the round 1 review probe), `wordmark` three thin
    lines in one corner."""
    x, y, w, h = box
    ink = list(ink)
    if kind == "wordmark":
        for i in range(3):
            img = img.draw_rect(
                ink, x + w // 10, y + h // 10 + i * h // 30, w // 4, h // 80, fill=True
            )
        return img
    img = img.draw_circle(ink, x + w // 2, y + h // 2, w // 3, fill=True)
    for i in range(4):
        img = img.draw_rect(
            ink, x + w // 8, y + h // 8 + i * h // 10, w * 3 // 4, h // 24, fill=True
        )
    return img


# --- providers that obey, half-obey or ignore the edit mask (T-38-1) -------------------------


def busy_scene(
    w: int, h: int, seed: int = 11, grain: float = 4.0, tone: float = 1.0
) -> pyvips.Image:
    """A deliberately hard background: gradients, texture bands, sensor-grain noise (`grain`
    levels sigma) and a few hard-edged shapes (nothing like the flat base background). `tone` scales the room's light
    (0.5 = a dark room)."""
    xyz = pyvips.Image.xyz(w, h)
    xx, yy = xyz[0].cast("float") / w, xyz[1].cast("float") / h
    r = 90 + 120 * xx + 25 * (yy * 5400).sin()
    g = 120 + 90 * yy + 25 * (xx * 4300).sin()
    b = 160 - 60 * xx * yy + 20 * ((xx + yy) * 3100).sin()
    img = r.bandjoin([g, b]) * tone + pyvips.Image.gaussnoise(w, h, sigma=grain, mean=0, seed=seed)
    for fx, fy, fw, fh, col in (
        (0.05, 0.05, 0.22, 0.16, [250, 250, 245]),
        (0.70, 0.10, 0.20, 0.30, [30, 60, 40]),
        (0.02, 0.75, 0.30, 0.20, [200, 60, 50]),
        (0.78, 0.78, 0.16, 0.16, [20, 20, 25]),
    ):
        img = img.draw_rect(
            col, round(fx * w), round(fy * h), round(fw * w), round(fh * h), fill=True
        )
    return (img > 255).ifthenelse(255, (img < 0).ifthenelse(0, img))


def keep_from_mask(base_png: bytes, mask_png: bytes) -> tuple[pyvips.Image, pyvips.Image]:
    base = pyvips.Image.new_from_buffer(base_png, "")[0:3]
    keep = pyvips.Image.new_from_buffer(mask_png, "")[3] > 127
    return base, keep


def obeys_mask(
    base_png: bytes, mask_png: bytes, gain: float = 1.0, scene: pyvips.Image | None = None
) -> pyvips.Image:
    """A provider that keeps every opaque mask pixel and draws a busy scene everywhere else."""
    base, keep = keep_from_mask(base_png, mask_png)
    scene = busy_scene(base.width, base.height) if scene is None else scene
    img = keep.ifthenelse(base.cast("float") * gain, scene)
    return img.cast("uchar").copy(interpretation="srgb")


def to_provider_size(img: pyvips.Image, size: tuple[int, int]) -> pyvips.Image:
    """The frame contained in a provider-size canvas (its sides outpainted with a busy scene)."""
    w, h = img.width, img.height
    ow, oh = size
    if (ow, oh) == (w, h):
        return img
    fit = min(ow / w, oh / h)
    canvas = busy_scene(ow, oh, seed=5).cast("uchar")
    placed = scenes._warp(img, fit, ((ow - w * fit) / 2, (oh - h * fit) / 2), ow, oh).cast("uchar")
    cx, cy = round((ow - w * fit) / 2), round((oh - h * fit) / 2)
    cw, ch = round(w * fit), round(h * fit)
    sel = pyvips.Image.black(ow, oh).draw_rect(255, cx, cy, cw, ch, fill=True)
    return sel.ifthenelse(placed, canvas).copy(interpretation="srgb")


def rerendered(img: pyvips.Image, jpeg_q: int = 85, noise: float = 3.0, blur: float = 0.5):
    """What an obedient provider still does to kept pixels: re-encode the whole frame."""
    out = img.cast("float")
    if noise:
        out = out + pyvips.Image.gaussnoise(img.width, img.height, sigma=noise, mean=0, seed=3)
    out = (out > 255).ifthenelse(255, (out < 0).ifthenelse(0, out)).cast("uchar")
    if blur:
        out = out.gaussblur(blur)
    return pyvips.Image.new_from_buffer(out.jpegsave_buffer(Q=jpeg_q), "").copy(
        interpretation="srgb"
    )


def old_mask_only(
    base_png: bytes,
    protected_box: tuple[int, int, int, int],
    garment: str,
    view: str,
    blank: str,
    shift_in: float = 0.0,
    widen: float = 1.0,
) -> pyvips.Image:
    """A provider that obeyed only the old mask: the print box + 0.5 in rectangle is untouched,
    the garment outside it is re-drawn (moved down `shift_in` on the unmoved figure, or widened)
    over a busy scene in place of the flat background."""
    base = pyvips.Image.new_from_buffer(base_png, "")[0:3]
    w, h = base.width, base.height
    redrawn = moved_garment(garment, view, w, h, blank, dy_in=shift_in, widen=widen)
    bg = scenes.LIFESTYLE_BG if view == "lifestyle_base" else (255, 255, 255)
    is_bg = (redrawn == list(bg)).bandand()
    img = is_bg.ifthenelse(busy_scene(w, h), redrawn.cast("float"))
    x, y, bw, bh = protected_box
    inside = pyvips.Image.black(w, h).draw_rect(255, x, y, bw, bh, fill=True) > 127
    return inside.ifthenelse(base.cast("float"), img).cast("uchar").copy(interpretation="srgb")
