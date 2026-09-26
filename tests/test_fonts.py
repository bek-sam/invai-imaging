import pytest

from app.fonts import known_family, missing_chars, normalize_family


def test_known_family_is_case_and_spacing_insensitive():
    assert known_family("Inter Bold")
    assert known_family("inter-bold")
    assert known_family("INTER_BOLD")
    assert known_family("Bebas Neue")


def test_unknown_family_is_not_known_and_normalize_raises():
    assert not known_family("Comic Sans")
    assert not known_family(None)
    with pytest.raises(ValueError, match="unknown font family"):
        normalize_family("Comic Sans")


def test_missing_chars_flags_glyphs_the_font_lacks():
    # Bebas Neue (Latin Extended-B display font) has no glyph for this Vietnamese-only letter.
    assert missing_chars("Bebas Neue", "ɂ") == ["ɂ"]
    # Ordinary Western names are fully covered by every bundled font.
    assert missing_chars("Bebas Neue", "José Núñez") == []
    assert missing_chars("Inter", "José Núñez") == []


def test_missing_chars_ignores_whitespace():
    assert missing_chars("Oswald", "  \t\n") == []
