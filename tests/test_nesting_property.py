"""Property tests for gang-sheet nesting (T-32-3, research 12 G22).

Random piece sets inside the limits: nothing overlaps (spacing respected), everything fits the
sheet, sizes are kept (rotation swaps w/h only), every copy appears exactly once, an item that
cannot fit raises NestError, the label strip is part of each footprint, utilization is 0..1.

Run with `uv run pytest tests/test_nesting_property.py`. `derandomize=True` keeps CI reproducible;
on a failure Hypothesis prints the minimal example and `@reproduce_failure` blob (print_blob).
Sizes are generated on the 0.001 in grid because nest() works in integer mils.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.nesting import Item, NestError, Placement, nest, overall_utilization

MILS = 1000
SETTINGS = settings(
    max_examples=120,
    deadline=None,
    derandomize=True,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)


def mils(v: float) -> int:
    return round(v * MILS)


@st.composite
def scenario(draw):
    width = draw(st.integers(8_000, 30_000)) / MILS
    max_len = draw(st.one_of(st.integers(8_000, 40_000), st.integers(40_000, 240_000))) / MILS
    spacing = draw(st.integers(0, 1_000)) / MILS
    margin = draw(st.integers(0, 1_000)) / MILS
    label = draw(st.integers(0, 1_000)) / MILS
    header = draw(st.integers(0, 1_000)) / MILS
    rotation = draw(st.booleans())
    # Mostly sizes that fit, with the odd oversize one so NestError is exercised too.
    dim = st.one_of(st.integers(500, 9_000), st.integers(500, 22_000))
    n = draw(st.integers(1, 8))
    items = [
        Item(f"i{k}", draw(dim) / MILS, draw(dim) / MILS, draw(st.integers(0, 12)))
        for k in range(n)
    ]
    return items, {
        "sheet_width_in": width,
        "spacing_in": spacing,
        "margin_in": margin,
        "max_length_in": max_len,
        "allow_rotation": rotation,
        "label_height_in": label,
        "header_height_in": header,
    }


def usable(kw):
    uw = mils(kw["sheet_width_in"]) - 2 * mils(kw["margin_in"])
    uh = mils(kw["max_length_in"]) - 2 * mils(kw["margin_in"]) - mils(kw["header_height_in"])
    return uw, uh


def fits(it: Item, kw) -> bool:
    uw, uh = usable(kw)
    w, h, lab = mils(it.width_in), mils(it.height_in), mils(kw["label_height_in"])
    ok = w <= uw and h + lab <= uh
    if kw["allow_rotation"]:
        ok = ok or (h <= uw and w + lab <= uh)
    return ok


def box(p: Placement, label_in: float):
    x, y = mils(p.x_in), mils(p.y_in)
    return x, y, x + mils(p.width_in), y + mils(p.height_in) + mils(label_in)


@SETTINGS
@given(scenario())
def test_nest_properties(sc):
    items, kw = sc
    uw, uh = usable(kw)
    live = [i for i in items if i.quantity >= 1]
    must_fail = uw <= 0 or uh <= 0 or any(not fits(i, kw) for i in live)
    try:
        sheets = nest(items, **kw)
    except NestError:
        assert must_fail, "NestError raised although every item fits"
        return
    assert not must_fail, "nest returned sheets although an item cannot fit"

    spacing, margin = mils(kw["spacing_in"]), mils(kw["margin_in"])
    header, width = mils(kw["header_height_in"]), mils(kw["sheet_width_in"])
    label = kw["label_height_in"]
    by_id = {i.id: i for i in live}
    seen: list[tuple[str, int]] = []

    for s in sheets:
        assert s.length_in <= kw["max_length_in"] + 1e-6
        assert 0 <= s.utilization <= 1
        for p in s.placements:
            seen.append((p.id, p.copy))
            x0, y0, x1, y1 = box(p, label)
            # inside the sheet, margins and header band kept clear
            assert x0 >= margin and x1 <= width - margin
            assert y0 >= margin + header
            assert y1 <= mils(s.length_in) - margin + 1
            # the footprint height includes the label strip
            assert y1 - y0 == mils(p.height_in) + mils(label)
            # size kept; rotation only swaps width and height
            src = by_id[p.id]
            sw, sh = mils(src.width_in), mils(src.height_in)
            want = (sh, sw) if p.rotated else (sw, sh)
            assert (mils(p.width_in), mils(p.height_in)) == want
            if not kw["allow_rotation"]:
                assert not p.rotated
        boxes = [box(p, label) for p in s.placements]
        for a_i, a in enumerate(boxes):
            for b in boxes[a_i + 1 :]:
                apart = (
                    a[2] + spacing <= b[0]
                    or b[2] + spacing <= a[0]
                    or a[3] + spacing <= b[1]
                    or b[3] + spacing <= a[1]
                )
                assert apart, f"overlap or gap < spacing: {a} vs {b}"

    expected = [(i.id, c) for i in live for c in range(i.quantity)]
    assert sorted(seen) == sorted(expected), "every copy must appear exactly once"
    assert 0 <= overall_utilization(sheets) <= 1


@SETTINGS
@given(
    st.integers(1, 20_000),
    st.integers(1, 20_000),
    st.integers(0, 600),
)
def test_oversized_item_raises(w, h, extra):
    """A design wider and taller than the film in both orientations can never be placed."""
    usable_w = 22_000 - 500
    too_big = Item("big", (usable_w + 1 + extra) / MILS, (usable_w + 1 + extra) / MILS)
    try:
        nest([Item("ok", w / MILS, h / MILS), too_big])
    except NestError:
        return
    raise AssertionError("an item wider than the film in both orientations must raise NestError")


@SETTINGS
@given(scenario())
def test_nest_is_deterministic(sc):
    items, kw = sc
    try:
        a = nest(items, **kw)
        b = nest(items, **kw)
    except NestError:
        return
    assert a == b


@SETTINGS
@given(st.integers(500, 9_000), st.integers(1, 4), st.booleans())
def test_exact_fit_boundary(h, qty, rotation):
    """A design exactly as wide as the usable film (22 in minus two 0.25 in margins) is placed;
    one mil wider cannot be (taller than the film too, so rotation cannot save it)."""
    usable_w = 21_500
    sheets = nest([Item("edge", usable_w / MILS, h / MILS, qty)], allow_rotation=rotation)
    assert sum(len(s.placements) for s in sheets) == qty
    too_wide = Item("wide", (usable_w + 1) / MILS, 300.0)
    try:
        nest([too_wide], allow_rotation=rotation)
    except NestError:
        return
    raise AssertionError("one mil past the usable width must raise NestError")
