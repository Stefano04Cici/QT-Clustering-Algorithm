import pytest

from qt_clustering.core.attributes import ContinuousAttribute
from qt_clustering.core.cluster import Cluster
from qt_clustering.core.data import Data
from qt_clustering.core.items import ContinuousItem, Item
from qt_clustering.core.tuple import QTuple
import numpy as np


def _make_cluster(centroid_vals: list[float], member_ids: set[int]) -> Cluster:
    items: list[Item] = [
        ContinuousItem(
            attribute=ContinuousAttribute(name=f"f{i}", index=i, min_val=0.0, max_val=10.0),
            value=v,
        )
        for i, v in enumerate(centroid_vals)
    ]
    centroid = QTuple(items=items)
    return Cluster(centroid=centroid, clustered_data=set(member_ids))


class TestCluster:
    def test_add_data(self):
        c = _make_cluster([5.0], set())
        assert c.add_data(0) is True
        assert c.get_size() == 1

    def test_add_duplicate(self):
        c = _make_cluster([5.0], {0})
        assert c.add_data(0) is False
        assert c.get_size() == 1

    def test_contains(self):
        c = _make_cluster([5.0], {0, 1})
        assert c.contains(0) is True
        assert c.contains(2) is False

    def test_iter(self):
        c = _make_cluster([5.0], {0, 1, 2})
        assert sorted(c) == [0, 1, 2]

    def test_ordering_by_size(self):
        c1 = _make_cluster([1.0], {0})
        c2 = _make_cluster([1.0], {0, 1, 2})
        assert c1 < c2

    def test_ordering_by_centroid(self):
        c1 = _make_cluster([1.0, 2.0], {0})
        c2 = _make_cluster([3.0, 2.0], {0})
        assert c1 < c2

    def test_to_string(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = Data.from_numpy(X, ["a", "b"])
        c = _make_cluster([1.0, 2.0], {0, 1})
        s = c.to_string(data)
        assert "Cluster" in s
        assert "Example 0" in s
