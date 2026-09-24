import pytest
import pyvips
from PIL import Image, ImageDraw

from app.qa import check, clean_alpha, effective_dpi, soft_alpha_ratio


def codes(res):
    return {(i.code, i.severity) for i in res.issues}


def test_effective_dpi_math():
    assert effective_dpi(3000, 3000, 10, None) == 300
    assert effective_dpi(3000, 3000, None, 12) == 250
    # the limiting axis wins
    assert effective_dpi(3000, 2000, 10, 10) == 200
    assert effective_dpi(3000, 2000, None, None) is None


def test_soft_alpha_ratio():
    hist = [0] * 256
    hist[0], hist[255], hist[128] = 1000, 90, 10
    assert soft_alpha_ratio(hist) == pytest.approx(0.1)


def test_low_dpi_levels(tmp_path):
    p = tmp_path / "a.png"
    im = Image.new("RGBA", (1000, 1000), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((100, 100, 900, 900), fill=(255, 0, 0, 255))
    im.save(p)
    assert ("low_dpi", "error") in codes(check(p, 10))  # 100 DPI
    assert ("low_dpi", "warn") in codes(check(p, 5))  # 200 DPI
    res = check(p, 3)
    assert res.effective_dpi == pytest.approx(333.3)
    assert not res.issues
    assert res.has_alpha and res.soft_alpha_ratio == 0


def test_no_alpha_and_soft_alpha(tmp_path):
    p = tmp_path / "rgb.jpg"
    Image.new("RGB", (800, 800), (200, 10, 10)).save(p)
    assert ("no_alpha", "warn") in codes(check(p))

    soft = tmp_path / "soft.png"
    Image.new("RGBA", (800, 800), (200, 10, 10, 120)).save(soft)
    res = check(soft)
    assert res.soft_alpha_ratio == 1.0
    assert ("soft_alpha", "warn") in codes(res)

    out = tmp_path / "clean.png"
    clean_alpha(soft, out, threshold=100)
    cleaned = check(out)
    assert cleaned.soft_alpha_ratio == 0
    assert pyvips.Image.new_from_file(str(out)).getpoint(5, 5)[3] == 255
    clean_alpha(soft, out, threshold=200)
    assert pyvips.Image.new_from_file(str(out)).getpoint(5, 5)[3] == 0


def test_unreadable_and_tiny(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    res = check(bad)
    assert ("unreadable", "error") in codes(res)

    tiny = tmp_path / "tiny.png"
    Image.new("RGBA", (40, 40), (0, 0, 0, 255)).save(tiny)
    assert ("tiny_file", "warn") in codes(check(tiny))
