"""Print-file checks for DTF: effective DPI, transparency, soft alpha; and alpha cleanup."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyvips

from app import qa_print
from app.vips import ImageError, load, save_png, to_srgba

DPI_ERROR = 150
DPI_WARN = 300
SOFT_ALPHA_WARN = 0.05  # anti-aliased edges alone are usually 1–3%
TINY_BYTES = 2_048
TINY_PX = 150


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    # Print-file checks only (T-42-4): inches, unrounded, and boxes as 0..1 ratios of the image.
    measured_in: float | None = None
    regions: list[dict[str, float]] | None = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "severity": self.severity, "message": self.message}
        if self.measured_in is not None:
            d["measured_in"] = self.measured_in
        if self.regions:
            d["regions"] = self.regions
        return d


@dataclass
class QAResult:
    width_px: int = 0
    height_px: int = 0
    has_alpha: bool = False
    soft_alpha_ratio: float = 0.0
    effective_dpi: float | None = None
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        if d["effective_dpi"] is None:
            del d["effective_dpi"]
        return d


def effective_dpi(
    width_px: int, height_px: int, target_width_in: float | None, target_height_in: float | None
) -> float | None:
    """DPI the file will have when printed at the target size (the limiting axis wins)."""
    dpis = []
    if target_width_in:
        dpis.append(width_px / target_width_in)
    if target_height_in:
        dpis.append(height_px / target_height_in)
    return round(min(dpis), 1) if dpis else None


def alpha_histogram(img: pyvips.Image) -> list[int]:
    alpha = to_srgba(img)[3]
    hist = alpha.hist_find()
    return list(memoryview(hist.write_to_memory()).cast("I"))


def soft_alpha_ratio(hist: list[int]) -> float:
    visible = sum(hist[1:])
    return sum(hist[1:255]) / visible if visible else 0.0


def check(
    path: Path,
    target_width_in: float | None = None,
    target_height_in: float | None = None,
    target_dpi: float | None = None,
) -> QAResult:
    res = QAResult()
    size = path.stat().st_size
    try:
        # PDF has no pixels until rasterized, and `load()` requires a target_dpi for it (T-9-1);
        # SVG's default (no target_dpi) renders at a 1:1 user-unit scale, not a print-quality
        # size. Vector art has no "native" resolution the way a raster file does, so QA
        # rasterizes it at the caller's intended print DPI, falling back to the shop's own
        # recommended DPI (DPI_WARN) when the caller doesn't say. Raster formats ignore this
        # (load() only uses target_dpi for pdf/svg), so passing it unconditionally is harmless.
        img = load(path, access="sequential", target_dpi=target_dpi or DPI_WARN)
        # Decode the pixels once so truncated files are caught here.
        hist = alpha_histogram(img)
    except (ImageError, pyvips.Error) as e:
        msg = str(e).splitlines()[0] if str(e) else "unreadable image"
        res.issues.append(Issue("unreadable", "error", msg))
        return res

    res.width_px, res.height_px = img.width, img.height
    res.has_alpha = bool(img.hasalpha())
    res.soft_alpha_ratio = round(soft_alpha_ratio(hist), 4) if res.has_alpha else 0.0
    res.effective_dpi = effective_dpi(img.width, img.height, target_width_in, target_height_in)

    if res.effective_dpi is not None:
        if res.effective_dpi < DPI_ERROR:
            res.issues.append(
                Issue(
                    "low_dpi",
                    "error",
                    f"{res.effective_dpi:.0f} DPI at the target size; at least {DPI_ERROR} is "
                    f"needed and {DPI_WARN} is recommended",
                )
            )
        elif res.effective_dpi < DPI_WARN:
            res.issues.append(
                Issue(
                    "low_dpi",
                    "warn",
                    f"{res.effective_dpi:.0f} DPI at the target size; {DPI_WARN} is recommended",
                )
            )

    fully_opaque = hist[255] == img.width * img.height
    if not res.has_alpha or fully_opaque:
        res.issues.append(
            Issue(
                "no_alpha",
                "warn",
                "no transparent background; the whole rectangle will print"
                if not res.has_alpha
                else "alpha channel is fully opaque; the whole rectangle will print",
            )
        )
    elif res.soft_alpha_ratio > SOFT_ALPHA_WARN:
        res.issues.append(
            Issue(
                "soft_alpha",
                "warn",
                f"{res.soft_alpha_ratio:.0%} of visible pixels are semi-transparent; they print "
                "as a faint haze on DTF. Run clean-alpha",
            )
        )

    if size < TINY_BYTES or min(img.width, img.height) < TINY_PX:
        res.issues.append(
            Issue(
                "tiny_file",
                "warn",
                f"file is very small ({img.width}x{img.height}px, {size} bytes)",
            )
        )

    # Print-file checks (T-42-4). The profile is read from the file as loaded, before any
    # conversion; the art checks need transparency, so a fully opaque or blank file skips them.
    if qa_print.is_non_srgb(img):
        res.issues.append(
            Issue(
                "non_srgb",
                "warn",
                "this file uses a color profile other than sRGB (Adobe RGB, Display P3 or CMYK), "
                "so colors may shift when printed",
            )
        )
    if res.has_alpha and hist[255] != img.width * img.height and sum(hist[1:]) > 0:
        try:
            res.issues.extend(
                _art_issues(
                    path, res, target_dpi or DPI_WARN, bool(target_width_in or target_height_in)
                )
            )
        except (ImageError, pyvips.Error):
            pass  # the first pass decoded the file; a second failure only skips these checks
    return res


def _art_issues(path: Path, res: QAResult, load_dpi: float, has_target: bool) -> list[Issue]:
    """thin_line, small_text and white_box from one shrunk copy of the art."""
    scale = qa_print.analysis_scale(res.width_px, res.height_px, res.effective_dpi)
    rgba = qa_print.analysis_array(
        to_srgba(load(path, access="sequential", target_dpi=load_dpi)), scale
    )
    out: list[Issue] = []
    if has_target and res.effective_dpi:
        adpi = res.effective_dpi * rgba.shape[1] / res.width_px
        shrunk = rgba.shape[1] < res.width_px
        cut = qa_print.ALPHA_VISIBLE_SHRUNK if shrunk else qa_print.ALPHA_VISIBLE
        mask = rgba[..., 3] >= cut
        thin = qa_print.thin_lines(mask, adpi)
        if thin:
            severity, width_in, regions = thin
            msg = (
                f"Some lines are too thin to print ({width_in:.3f} in)."
                if severity == "error"
                else f"Some lines are very thin ({width_in:.3f} in) and may not hold."
            )
            out.append(Issue("thin_line", severity, msg, width_in, regions))
        text = qa_print.small_text(mask, adpi)
        if text:
            height_in, regions = text
            out.append(
                Issue(
                    "small_text",
                    "warn",
                    "Small text may fill in when printed. Make it bigger if you can.",
                    height_in,
                    regions,
                )
            )
    box = qa_print.white_box(rgba)
    if box:
        out.append(
            Issue(
                "white_box",
                "warn",
                "A white box surrounds the art and will print on dark shirts. "
                "Remove the white background.",
                None,
                box,
            )
        )
    return out


def clean_alpha(src: Path, dest: Path, threshold: int = 128) -> None:
    """Snap alpha to 0/255 (>= threshold becomes opaque) so DTF gets no half-printed pixels."""
    if not 1 <= threshold <= 255:
        raise ImageError("threshold must be between 1 and 255")
    img = to_srgba(load(src, access="sequential"))
    rgb, alpha = img.extract_band(0, n=3), img[3]
    alpha = (alpha >= threshold).ifthenelse(255, 0).cast("uchar")
    out = rgb.bandjoin(alpha).copy(interpretation="srgb", xres=img.xres, yres=img.yres)
    save_png(out, dest)
