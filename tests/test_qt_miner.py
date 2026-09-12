import numpy as np
import pytest

from qt_clustering.core.data import Data
from qt_clustering.core.qt_miner import QTMiner
from qt_clustering.exceptions import ClusteringRadiusException, EmptyDatasetException


class TestQTMiner:
    def test_empty_dataset(self):
        data = Data.from_numpy(np.array([]).reshape(0, 3))
        miner = QTMiner(radius=0.5)
        with pytest.raises(EmptyDatasetException):
            miner.compute(data)

    def test_two_distinct_clusters(self):
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(20, 2))
        cluster2 = rng.normal(loc=10.0, scale=0.1, size=(20, 2))
        X = np.vstack([cluster1, cluster2])
        data = Data.from_numpy(X)

        miner = QTMiner(radius=0.5)
        n = miner.compute(data)
        assert n == 2

    def test_single_cluster_valid(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=0.01, size=(10, 2))
        data = Data.from_numpy(X)

        miner = QTMiner(radius=5.0)
        n = miner.compute(data)
        assert n == 1

        assigned = set()
        for cluster in miner.cluster_set:
            assigned.update(cluster)
        assert assigned == set(range(10))

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            QTMiner(radius=0.0)
        with pytest.raises(ValueError):
            QTMiner(radius=-1.0)

    def test_all_members_assigned(self):
        rng = np.random.default_rng(0)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(10, 2))
        cluster2 = rng.normal(loc=5.0, scale=0.1, size=(10, 2))
        X = np.vstack([cluster1, cluster2])
        data = Data.from_numpy(X)

        miner = QTMiner(radius=0.5)
        miner.compute(data)

        assigned = set()
        for cluster in miner.cluster_set:
            assigned.update(cluster)
        assert assigned == set(range(20))

    def test_no_overlap(self):
        rng = np.random.default_rng(1)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(15, 3))
        cluster2 = rng.normal(loc=5.0, scale=0.1, size=(15, 3))
        cluster3 = rng.normal(loc=10.0, scale=0.1, size=(15, 3))
        X = np.vstack([cluster1, cluster2, cluster3])
        data = Data.from_numpy(X)

        miner = QTMiner(radius=0.5)
        n = miner.compute(data)
        assert n == 3

        all_ids = []
        for cluster in miner.cluster_set:
            all_ids.append(sorted(cluster))
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                assert set(all_ids[i]).isdisjoint(set(all_ids[j]))

    def test_make_blobs_dataset(self):
        pytest.importorskip("sklearn")

        from sklearn.datasets import make_blobs

        blobs_result = make_blobs(
            n_samples=300,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            center_box=(-10.0, 10.0),
            random_state=42,
        )
        X = blobs_result[0]
        data = Data.from_numpy(X, [f"f{i}" for i in range(X.shape[1])])

        miner = QTMiner(radius=0.5)
        n = miner.compute(data)
        assert n > 1

        assigned = set()
        for cluster in miner.cluster_set:
            assigned.update(cluster)
        assert assigned == set(range(X.shape[0]))
