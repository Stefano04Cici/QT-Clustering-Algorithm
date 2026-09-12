from qt_clustering.core.attributes import ContinuousAttribute
from qt_clustering.core.cluster import Cluster
from qt_clustering.core.cluster_set import ClusterSet
from qt_clustering.core.items import ContinuousItem, Item
from qt_clustering.core.tuple import QTuple


def _make_cluster(size: int, centroid_val: float) -> Cluster:
    items: list[Item] = [ContinuousItem(
        attribute=ContinuousAttribute(name="f", index=0, min_val=0.0, max_val=10.0),
        value=centroid_val,
    )]
    centroid = QTuple(items=items)
    return Cluster(centroid=centroid, clustered_data=set(range(size)))


class TestClusterSet:
    def test_add_and_len(self):
        cs = ClusterSet()
        cs.add(_make_cluster(3, 1.0))
        cs.add(_make_cluster(1, 2.0))
        assert len(cs) == 2

    def test_ordering(self):
        cs = ClusterSet()
        cs.add(_make_cluster(1, 5.0))
        cs.add(_make_cluster(5, 1.0))
        cs.add(_make_cluster(3, 2.0))
        sizes = [c.get_size() for c in cs]
        assert sizes == [1, 3, 5]

    def test_iter(self):
        cs = ClusterSet()
        cs.add(_make_cluster(2, 1.0))
        cs.add(_make_cluster(4, 2.0))
        sizes = [c.get_size() for c in cs]
        assert sizes == [2, 4]

    def test_getitem(self):
        cs = ClusterSet()
        cs.add(_make_cluster(1, 1.0))
        cs.add(_make_cluster(2, 2.0))
        assert cs[0].get_size() == 1
        assert cs[1].get_size() == 2

    def test_get_number_of_clusters(self):
        cs = ClusterSet()
        assert cs.get_number_of_clusters() == 0
        cs.add(_make_cluster(1, 1.0))
        assert cs.get_number_of_clusters() == 1

    def test_get_labels(self):
        cs = ClusterSet()
        c1 = _make_cluster(2, 1.0)
        c1.clustered_data = {0, 1}
        c2 = _make_cluster(3, 2.0)
        c2.clustered_data = {2, 3, 4}
        cs.add(c1)
        cs.add(c2)
        labels = cs.get_labels(5)
        assert labels.tolist() == [0, 0, 1, 1, 1]
