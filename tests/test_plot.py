import math
from typing import cast
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from qt_clustering.plot import generate_colors, plot_before_after
from qt_clustering.core.cluster import Cluster
from qt_clustering.core.cluster_set import ClusterSet
from qt_clustering.core.attributes import ContinuousAttribute
from qt_clustering.core.items import ContinuousItem, Item
from qt_clustering.core.tuple import QTuple


class TestGenerateColors:
    def test_count(self):
        assert len(generate_colors(5)) == 5

    def test_zero(self):
        assert generate_colors(0) == []

    def test_negative(self):
        assert generate_colors(-1) == []

    def test_range(self):
        for r, g, b in generate_colors(20):
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_single(self):
        colors = generate_colors(1)
        assert len(colors) == 1
        assert isinstance(colors[0], tuple)
        assert len(colors[0]) == 3

    def test_distinct_adjacent(self):
        colors = generate_colors(6)
        for i in range(len(colors) - 1):
            c1 = colors[i]
            c2 = colors[i + 1]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
            assert dist > 0.3, f"Colors {i} and {i + 1} too similar (dist={dist:.4f})"


class TestPlotBeforeAfter:
    def _make_two_clusters(self):
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=0.0, scale=0.1, size=(10, 2))
        c2 = rng.normal(loc=5.0, scale=0.1, size=(10, 2))
        X = np.vstack([c1, c2])
        cs = ClusterSet()
        items1 = cast(list[Item], [
            ContinuousItem(attribute=ContinuousAttribute(name="f0", index=0, min_val=0, max_val=5), value=0.0),
            ContinuousItem(attribute=ContinuousAttribute(name="f1", index=1, min_val=0, max_val=5), value=0.0),
        ])
        items2 = cast(list[Item], [
            ContinuousItem(attribute=ContinuousAttribute(name="f0", index=0, min_val=0, max_val=5), value=5.0),
            ContinuousItem(attribute=ContinuousAttribute(name="f1", index=1, min_val=0, max_val=5), value=5.0),
        ])
        cs.add(Cluster(centroid=QTuple(items=items1), clustered_data=set(range(10))))
        cs.add(Cluster(centroid=QTuple(items=items2), clustered_data=set(range(10, 20))))
        return X, cs

    @patch("qt_clustering.plot.plt")
    def test_plot_no_exception(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        X, cs = self._make_two_clusters()
        plot_before_after(X, cs)
        mock_plt.subplots.assert_called_once()
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()
