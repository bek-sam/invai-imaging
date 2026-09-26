import pytest

from app.nesting import Item, NestError, nest

LABEL = 0.42
SPACING = 0.25
MARGIN = 0.25
HEADER = 0.45
EPS = 1e-6


def footprints(sheet, spacing=SPACING, label=LABEL):
    """Design + label strip boxes, each grown by half the spacing on every side."""
    return [
        (
            p.x_in - spacing / 2,
            p.y_in - spacing / 2,
            p.x_in + p.width_in + spacing / 2,
            p.y_in + p.height_in + label + spacing / 2,
        )
        for p in sheet.placements
    ]


def assert_valid(
    sheets, width=22.0, max_len=240.0, spacing=SPACING, label=LABEL, margin=MARGIN, header=HEADER
):
    for s in sheets:
        for p in s.placements:
            assert p.x_in >= margin - EPS
            assert p.y_in >= margin + header - EPS
            assert p.x_in + p.width_in <= width - margin + EPS
            assert p.y_in + p.height_in + label <= s.length_in - margin + EPS
        assert s.length_in <= max_len + EPS
        boxes = footprints(s, spacing, label)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                apart = (
                    a[2] <= b[0] + EPS
                    or b[2] <= a[0] + EPS
                    or a[3] <= b[1] + EPS
                    or b[3] <= a[1] + EPS
                )
                assert apart, (a, b)


def test_mixed_items_no_overlap_within_width():
    items = [Item("a", 10, 12), Item("b", 4, 4), Item("c", 12, 3), Item("d", 3, 3, quantity=5)]
    sheets = nest(items)
    assert len(sheets) == 1
    assert len(sheets[0].placements) == 8
    assert_valid(sheets)


def test_quantity_expansion_and_copy_numbers():
    sheets = nest([Item("x", 5, 5, quantity=4)])
    copies = sorted(p.copy for p in sheets[0].placements)
    assert copies == [0, 1, 2, 3]
    assert {p.id for p in sheets[0].placements} == {"x"}


def test_true_used_length_not_bin_height():
    sheets = nest([Item("a", 10, 10)])
    # margin + header + design + label + margin; no trailing spacing
    assert sheets[0].length_in == pytest.approx(MARGIN + HEADER + 10 + LABEL + MARGIN)
    assert sheets[0].utilization == pytest.approx(100 / (22 * sheets[0].length_in), abs=1e-4)


def test_spacing_respected_exactly_side_by_side():
    sheets = nest([Item("a", 10, 10), Item("b", 10, 10)], allow_rotation=False)
    a, b = sorted(sheets[0].placements, key=lambda p: p.x_in)
    assert a.y_in == b.y_in
    assert b.x_in - (a.x_in + a.width_in) == pytest.approx(SPACING)


def test_rotation_used_when_it_is_the_only_fit():
    # 21.8in is wider than the 21.5in between the margins; rotated it fits.
    sheets = nest([Item("wide", 21.8, 4)])
    p = sheets[0].placements[0]
    assert p.rotated
    assert (p.width_in, p.height_in) == (4, 21.8)
    assert_valid(sheets)


def test_no_rotation_when_disabled():
    sheets = nest([Item(str(i), 3, 9, quantity=3) for i in range(5)], allow_rotation=False)
    assert not any(p.rotated for s in sheets for p in s.placements)
    assert_valid(sheets)


def test_multi_sheet_split():
    sheets = nest([Item("big", 10, 10, quantity=20)], max_length_in=60)
    assert len(sheets) > 1
    assert sum(len(s.placements) for s in sheets) == 20
    assert [s.index for s in sheets] == list(range(len(sheets)))
    assert_valid(sheets, max_len=60)


def test_item_too_big_raises():
    with pytest.raises(NestError):
        nest([Item("huge", 30, 30)])


def test_utilization_reasonable_for_realistic_batch():
    sizes = [(11, 11), (10, 12), (4, 4), (12, 3), (3.5, 3.5), (9, 11), (5, 7)]
    items = [Item(str(i), *sizes[i % len(sizes)], quantity=1 + i % 3) for i in range(30)]
    sheets = nest(items)
    assert_valid(sheets)
    assert sheets[0].utilization > 0.7


def test_adult_fronts_go_two_up_on_22in_film():
    # 10.5 x 12 in fronts fit two across only when kept upright; rotating them wastes a column.
    sheets = nest([Item("front", 10.5, 12, quantity=16)])
    assert_valid(sheets)
    assert len(sheets) == 1
    assert sheets[0].length_in < 110
    assert sheets[0].utilization >= 0.85


def test_realistic_size_mix_nests_dense():
    items = [
        Item("adult", 10.5, 12, quantity=22),
        Item("back", 12, 14, quantity=4),
        Item("chest", 3.75, 3.75, quantity=3),
        Item("sleeve", 3, 10, quantity=3),
        Item("youth", 8.5, 9.5, quantity=3),
    ]
    sheets = nest(items)
    assert_valid(sheets)
    assert sheets[0].utilization >= 0.8
