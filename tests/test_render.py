import pyvips

from app.render import Slot, render_personalization


def flags_for(res, slot):
    return {f.code for f in res.flags if f.slot == slot}


def test_shrinks_to_fit_then_flags_overflow(tmp_path):
    slots = [
        Slot("name", 0.5, 0.5, 4.5, 2, font_family="Oswald", font_size_pt=110),
        Slot("fits", 0.5, 3, 9, 2, font_size_pt=60),
        Slot("long", 0.5, 5.5, 3, 1, font_size_pt=72, max_chars=10),
        Slot("empty", 0.5, 7, 3, 1),
        Slot("odd", 0.5, 8, 9, 1, font_size_pt=24),
    ]
    values = {
        "name": "JACKSON",  # 5.6in wide at 110pt; fits the 4.5in slot at ~80%
        "fits": "Ava",
        "long": "Maximiliana Rodriguez",
        "odd": "Grandma 💖",
    }
    out = tmp_path / "p.png"
    res = render_personalization(10, 9.5, slots, values, 300, out)
    assert (res.width_px, res.height_px) == (3000, 2850)
    assert flags_for(res, "name") == set()
    assert flags_for(res, "fits") == set()
    assert flags_for(res, "long") == {"too_long", "overflow"}
    assert flags_for(res, "empty") == {"empty"}
    assert "suspicious_chars" in flags_for(res, "odd")

    img = pyvips.Image.new_from_file(str(out))
    assert (img.width, img.height, img.bands) == (3000, 2850, 4)
    # something was drawn inside the "name" slot, nothing in the empty one
    name = img.crop(150, 150, 1350, 600)[3]
    assert name.max() == 255
    # shrunk text stays inside its slot: nothing right of it on that row
    assert img.crop(1500, 150, 1500, 600)[3].max() == 0
    assert img.crop(150, 2100, 900, 300)[3].max() == 0
