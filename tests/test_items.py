import pytest

from qt_clustering.core.attributes import ContinuousAttribute, DiscreteAttribute
from qt_clustering.core.items import ContinuousItem, DiscreteItem


class TestContinuousItem:
    def test_distance_same(self):
        attr = ContinuousAttribute(name="x", index=0, min_val=0.0, max_val=10.0)
        item = ContinuousItem(attribute=attr, value=5.0)
        assert item.distance(5.0) == pytest.approx(0.0)

    def test_distance_opposite(self):
        attr = ContinuousAttribute(name="x", index=0, min_val=0.0, max_val=10.0)
        item = ContinuousItem(attribute=attr, value=0.0)
        assert item.distance(10.0) == pytest.approx(1.0)

    def test_distance_normalized(self):
        attr = ContinuousAttribute(name="x", index=0, min_val=0.0, max_val=10.0)
        item = ContinuousItem(attribute=attr, value=2.0)
        assert item.distance(8.0) == pytest.approx(0.6)


class TestDiscreteItem:
    def test_distance_equal(self):
        attr = DiscreteAttribute(name="color", index=0, values=frozenset({"red", "blue"}))
        item = DiscreteItem(attribute=attr, value="red")
        assert item.distance("red") == 0.0

    def test_distance_different(self):
        attr = DiscreteAttribute(name="color", index=0, values=frozenset({"red", "blue"}))
        item = DiscreteItem(attribute=attr, value="red")
        assert item.distance("blue") == 1.0
