"""Unit tests for app.vips helpers (T-P1-2, B-103: ICC-aware sRGB conversion)."""

from pathlib import Path

import pyvips
from PIL import Image, ImageCms

from app.vips import to_srgba


def test_cmyk_without_profile_is_converted_not_rejected(tmp_path: Path):
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (40, 40), (0, 255, 255, 0)).save(path)  # a red-ish color in RGB
    img = pyvips.Image.jpegload(str(path))
    assert img.interpretation == "cmyk"
    out = to_srgba(img)
    assert out.interpretation == "srgb"
    assert out.bands == 4
    r, g, b, a = out.getpoint(20, 20)
    assert r > g and r > b  # converted to a plausible RGB color, not rejected
    assert a == 255


def test_embedded_srgb_profile_round_trips_colors(tmp_path: Path):
    path = tmp_path / "rgb-icc.jpg"
    profile = ImageCms.createProfile("sRGB")
    icc_bytes = ImageCms.ImageCmsProfile(profile).tobytes()
    Image.new("RGB", (40, 40), (200, 50, 25)).save(path, icc_profile=icc_bytes)
    img = pyvips.Image.jpegload(str(path))
    assert img.get_typeof("icc-profile-data") != 0
    out = to_srgba(img)
    r, g, b, a = out.getpoint(20, 20)
    assert (round(r), round(g), round(b)) == (200, 50, 25)
    assert a == 255


def test_image_without_an_embedded_profile_is_unchanged(tmp_path: Path):
    path = tmp_path / "plain.png"
    Image.new("RGBA", (40, 40), (10, 20, 30, 128)).save(path)
    img = pyvips.Image.pngload(str(path))
    assert img.get_typeof("icc-profile-data") == 0
    out = to_srgba(img)
    assert out.getpoint(20, 20) == [10, 20, 30, 128]


def test_unparseable_embedded_profile_does_not_fail_the_conversion(tmp_path: Path):
    path = tmp_path / "bad-icc.jpg"
    Image.new("RGB", (40, 40), (10, 20, 30)).save(path)
    img = pyvips.Image.jpegload(str(path)).copy()
    # Graft a garbage "icc-profile-data" field so icc_transform's embedded=True hits a profile
    # it can't parse, instead of finding none at all; this must not raise ImageError/pyvips.Error.
    img.set_type(pyvips.GValue.blob_type, "icc-profile-data", b"not a real icc profile")
    out = to_srgba(img)
    assert out.interpretation == "srgb"
    r, g, b, _ = out.getpoint(20, 20)
    assert (round(r), round(g), round(b)) == (10, 20, 30)
