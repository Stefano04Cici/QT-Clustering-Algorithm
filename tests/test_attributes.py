import pytest

from qt_clustering.core.attributes import ContinuousAttribute, DiscreteAttribute


class TestContinuousAttribute:
    def test_create(self):
        attr = ContinuousAttribute(name="age", index=0, min_val=0.0, max_val=100.0)
        assert attr.name == "age"
        assert attr.index == 0
        assert attr.min_val == 0.0
        assert attr.max_val == 100.0

    def test_scaled_value_mid(self):
        attr = ContinuousAttribute(name="age", index=0, min_val=0.0, max_val=100.0)
        assert attr.get_scaled_value(50.0) == pytest.approx(0.5)

    def test_scaled_value_min(self):
        attr = ContinuousAttribute(name="age", index=0, min_val=0.0, max_val=100.0)
        assert attr.get_scaled_value(0.0) == pytest.approx(0.0)

    def test_scaled_value_max(self):
        attr = ContinuousAttribute(name="age", index=0, min_val=0.0, max_val=100.0)
        assert attr.get_scaled_value(100.0) == pytest.approx(1.0)

    def test_scaled_value_zero_range(self):
        attr = ContinuousAttribute(name="const", index=0, min_val=5.0, max_val=5.0)
        assert attr.get_scaled_value(5.0) == 0.0

    def test_frozen(self):
        attr = ContinuousAttribute(name="x", index=0, min_val=0.0, max_val=1.0)
        with pytest.raises(AttributeError):
            setattr(attr, "min_val", 10.0)


class TestDiscreteAttribute:
    def test_create(self):
        attr = DiscreteAttribute(name="color", index=0, values=frozenset({"red", "blue"}))
        assert attr.name == "color"
        assert len(attr) == 2

    def test_len(self):
        attr = DiscreteAttribute(name="color", index=0, values=frozenset({"a", "b", "c"}))
        assert len(attr) == 3
