from __future__ import annotations

import gc
import threading

import numpy as np
import pytest

from src.core.qt_miner import QTMiner

POINTS = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]


def partition(clusters):
    return sorted(sorted(int(i) for i in cluster) for cluster in clusters)


def reference_qt(points, radius):
    data = np.asarray(points, dtype=float)
    n_rows, n_cols = data.shape
    lower = data.min(axis=0)
    upper = data.max(axis=0)
    span = upper - lower

    scaled = [
        [
            (data[row, col] - lower[col]) / span[col] if span[col] else 0.0
            for col in range(n_cols)
        ]
        for row in range(n_rows)
    ]

    def distance(a, b):
        total = 0.0
        for value_a, value_b in zip(scaled[a], scaled[b]):
            total += abs(value_a - value_b)
        return total

    assigned = set()
    clusters = []
    while len(assigned) < n_rows:
        best = None
        for i in range(n_rows):
            if i in assigned:
                continue
            candidate = [i]
            for j in range(n_rows):
                if j in assigned or j == i:
                    continue
                if distance(i, j) <= radius:
                    candidate.append(j)
            if best is None or len(candidate) > len(best):
                best = candidate
        if best is None:
            raise AssertionError("No cluster could be formed from the remaining rows")

        clusters.append(best)
        assigned.update(best)

    return sorted(sorted(cluster) for cluster in clusters)


class TestRadius:
    def test_radius_can_be_changed(self):
        qt = QTMiner(0.75)
        qt.radius = 1.5
        assert qt.radius == 1.5

    @pytest.mark.parametrize("radius", [0.0, -0.1, -10.0])
    def test_invalid_radius(self, radius):
        with pytest.raises(ValueError):
            QTMiner(radius)

    def test_invalid_radius_on_assignment_keeps_previous_value(self):
        qt = QTMiner(1.0)
        with pytest.raises(ValueError):
            qt.radius = 0.0
        assert qt.radius == 1.0


class TestCompute:
    def test_returns_number_of_clusters(self):
        qt = QTMiner(0.5)
        assert qt.compute(POINTS) == 2

    def test_clusters(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        assert qt.clusters == [[0, 1], [2, 3]]
        assert qt.cluster_count == 2
        assert qt.get_number_of_clusters() == 2
        assert len(qt) == 2

    def test_single_cluster_with_large_radius(self):
        qt = QTMiner(100.0)
        assert qt.compute(POINTS) == 1
        assert qt.clusters == [[0, 1, 2, 3]]
        assert qt.outlier_percentage() == 0.0

    def test_all_singletons_with_tiny_radius(self):
        qt = QTMiner(0.0001)
        assert qt.compute(POINTS) == 4
        assert [len(cluster) for cluster in qt.clusters] == [1, 1, 1, 1]
        assert qt.outlier_percentage() == 100.0

    def test_single_example(self):
        qt = QTMiner(1.0)
        assert qt.compute([[5.0, 5.0]]) == 1
        assert qt.clusters == [[0]]
        assert qt.outlier_percentage() == 100.0

    def test_clusters_are_sorted_by_size(self):
        points = []
        for base, count in ((0.0, 4), (10.0, 3), (20.0, 2)):
            points.extend([[base + 0.01 * step, 0.0] for step in range(count)])
        qt = QTMiner(0.05)
        assert qt.compute(points) == 3
        assert [len(cluster) for cluster in qt.clusters] == [2, 3, 4]

    def test_repeated_compute_accumulates(self):
        qt = QTMiner(0.5)
        assert qt.compute(POINTS) == 2
        assert qt.compute(POINTS) == 4
        assert len(qt.clusters) == 4

    def test_elapsed_ms_is_measured(self):
        rng = np.random.default_rng(0)
        qt = QTMiner(0.05)
        qt.compute(rng.random((60, 3)))
        assert qt.elapsed_ms > 0.0

    def test_str_contains_clusters(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS, feature_names=["x", "y"])
        text = str(qt)
        assert text.startswith("ClusterSet(2 clusters):")
        assert "Cluster(size=2" in text
        assert "x=" in text and "y=" in text

    def test_repr(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        assert repr(qt) == "QTMiner(radius=0.500000, clusters=2)"

    def test_clusters_stay_valid_after_garbage_collection(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        gc.collect()
        assert "feat_0" in str(qt)
        assert qt.clusters == [[0, 1], [2, 3]]

    def test_matches_reference_implementation(self):
        rng = np.random.default_rng(42)
        points = np.vstack(
            [
                rng.normal(0.0, 0.1, size=(7, 3)),
                rng.normal(5.0, 0.1, size=(6, 3)),
                rng.normal(10.0, 0.1, size=(5, 3)),
            ]
        )
        for radius in (0.01, 0.05, 0.5, 1.5, 10.0):
            qt = QTMiner(radius)
            qt.compute(points)
            assert partition(qt.clusters) == reference_qt(points, radius)


class TestLabelsAndOutliers:
    def test_labels_shape_and_dtype(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        labels = qt.labels(len(POINTS))
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (len(POINTS),)
        assert labels.dtype == np.int32

    def test_labels_values(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        assert qt.labels(len(POINTS)).tolist() == [0, 0, 1, 1]

    def test_labels_match_clusters(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        labels = qt.labels(len(POINTS))
        assert (labels >= 0).all()
        for cluster_index, cluster in enumerate(qt.clusters):
            for sample_index in cluster:
                assert labels[sample_index] == cluster_index

    def test_get_labels_is_an_alias(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS)
        assert qt.get_labels(len(POINTS)).tolist() == qt.labels(len(POINTS)).tolist()

    def test_outlier_percentage(self):
        points = [[0.0, 0.0], [0.01, 0.01], [5.0, 5.0], [9.0, 9.0], [13.0, 13.0]]
        qt = QTMiner(0.05)
        assert qt.compute(points) == 4
        assert qt.outlier_percentage() == pytest.approx(60.0)


class TestInputFormats:
    def test_list_of_lists(self):
        qt = QTMiner(0.5)
        assert qt.compute(POINTS) == 2

    def test_numpy_float(self):
        qt = QTMiner(0.5)
        assert qt.compute(np.asarray(POINTS, dtype=float)) == 2

    def test_numpy_integer(self):
        qt = QTMiner(0.5)
        assert qt.compute(np.array([[0, 0], [10, 10], [20, 20]])) == 3

    def test_feature_names(self):
        qt = QTMiner(0.5)
        qt.compute(POINTS, feature_names=["alpha", "beta"])
        text = str(qt)
        assert "alpha=" in text and "beta=" in text

    def test_csv_path_string(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text("x,y\n0,0\n0.1,0.1\n10,10\n10.1,10.1\n")
        qt = QTMiner(0.5)
        assert qt.compute(str(path)) == 2
        assert qt.clusters == [[0, 1], [2, 3]]

    def test_csv_path_object(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text("x,y\n0,0\n0.1,0.1\n10,10\n10.1,10.1\n")
        qt = QTMiner(0.5)
        assert qt.compute(path) == 2

    def test_csv_with_categorical_column(self, tmp_path):
        path = tmp_path / "mixed.csv"
        path.write_text(
            "age,color,score\n25,red,1.5\n30,blue,2.5\n40,red,3.5\n"
        )
        qt = QTMiner(0.5)
        assert qt.compute(path) == 3
        assert "color=red" in str(qt)

    def test_csv_with_quoted_fields(self, tmp_path):
        path = tmp_path / "quoted.csv"
        path.write_text('name,value\n"a,b",1.0\n"c,d",2.0\n')
        qt = QTMiner(0.5)
        assert qt.compute(path) == 2
        assert "a,b" in str(qt)


class TestErrors:
    def test_empty_matrix(self):
        with pytest.raises(RuntimeError):
            QTMiner(1.0).compute(np.zeros((0, 3)))

    def test_one_dimensional_input(self):
        with pytest.raises(ValueError):
            QTMiner(1.0).compute([1.0, 2.0])

    def test_ragged_input(self):
        with pytest.raises((TypeError, ValueError)):
            QTMiner(1.0).compute([[1.0, 2.0], [3.0]])

    def test_non_numeric_matrix(self):
        with pytest.raises((TypeError, ValueError)):
            QTMiner(1.0).compute([["a", "b"], ["c", "d"]])

    def test_missing_csv_file(self):
        with pytest.raises(RuntimeError):
            QTMiner(1.0).compute("this-file-does-not-exist.csv")

    def test_too_few_feature_names(self):
        with pytest.raises(IndexError):
            QTMiner(1.0).compute(POINTS, feature_names=["only_one"])


class TestConcurrency:
    def test_parallel_compute(self):
        results = []
        lock = threading.Lock()

        def worker():
            qt = QTMiner(0.5)
            count = qt.compute(POINTS)
            text = str(qt)
            with lock:
                results.append((count, partition(qt.clusters), "feat_0" in text))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results == [(2, [[0, 1], [2, 3]], True)] * 8
