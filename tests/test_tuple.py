import pytest

from qt_clustering.core.attributes import ContinuousAttribute
from qt_clustering.core.items import ContinuousItem, Item
from qt_clustering.core.tuple import QTuple


def _make_tuple(values: list[float]) -> QTuple:
    items: list[Item] = [
        ContinuousItem(
            attribute=ContinuousAttribute(name=f"f{i}", index=i, min_val=0.0, max_val=10.0),
            value=v,
        )
        for i, v in enumerate(values)
    ]
    return QTuple(items=items)


class TestQTuple:
    def test_get_length(self):
        t = _make_tuple([1.0, 2.0, 3.0])
        assert t.get_length() == 3

    def test_get_item(self):
        t = _make_tuple([1.0, 2.0])
        assert t.get(0).value == 1.0
        assert t.get(1).value == 2.0

    def test_distance_identical(self):
        t1 = _make_tuple([1.0, 2.0])
        t2 = _make_tuple([1.0, 2.0])
        assert t1.get_distance(t2) == pytest.approx(0.0)

    def test_distance_different(self):
        t1 = _make_tuple([0.0, 0.0])
        t2 = _make_tuple([10.0, 10.0])
        assert t1.get_distance(t2) == pytest.approx(2.0)

    def test_distance_partial(self):
        t1 = _make_tuple([0.0, 5.0])
        t2 = _make_tuple([10.0, 5.0])
        assert t1.get_distance(t2) == pytest.approx(1.0)
