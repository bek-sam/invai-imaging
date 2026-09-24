from app.nesting import Item, nest, utilization


def test_all_items_fit_within_width():
    items = [Item(str(i), 11.0, 11.0) for i in range(8)]
    placements, length = nest(items)
    assert len(placements) == 8
    assert all(p.x_in + p.width_in <= 22.0 for p in placements)
    assert length > 0
    assert 0 < utilization(items, length) <= 1


def test_no_overlap():
    items = [Item("a", 10, 12), Item("b", 4, 4), Item("c", 12, 3), Item("d", 3, 3)]
    placements, _ = nest(items)
    for i, a in enumerate(placements):
        for b in placements[i + 1 :]:
            separate = (
                a.x_in + a.width_in <= b.x_in
                or b.x_in + b.width_in <= a.x_in
                or a.y_in + a.height_in <= b.y_in
                or b.y_in + b.height_in <= a.y_in
            )
            assert separate, (a, b)
