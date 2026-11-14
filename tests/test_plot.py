from __future__ import annotations

import colorsys

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

import src.plot as plot_module
from src.core.qt_miner import QTMiner

POINTS = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
FEATURE_NAMES = ["x", "y"]
STEELBLUE = (0.27450980392156865, 0.5098039215686274, 0.7058823529411765)


@pytest.fixture
def shown_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: figures.append(plt.gcf())
    )
    yield figures
    plt.close("all")


def make_miner(radius: float) -> QTMiner:
    miner = QTMiner(radius)
    miner.compute(POINTS, feature_names=FEATURE_NAMES)
    return miner


class TestGenerateColors:
    @pytest.mark.parametrize("count", [0, -1, -10])
    def test_non_positive_count_returns_empty_list(self, count):
        assert plot_module.generate_colors(count) == []

    @pytest.mark.parametrize("count", [1, 2, 5, 17])
    def test_returns_requested_number_of_rgb_colors(self, count):
        colors = plot_module.generate_colors(count)
        assert len(colors) == count
        assert all(len(color) == 3 for color in colors)
        assert all(0.0 <= channel <= 1.0 for color in colors for channel in color)

    def test_is_deterministic(self):
        assert plot_module.generate_colors(8) == plot_module.generate_colors(8)

    def test_uses_golden_angle_hues(self):
        colors = plot_module.generate_colors(6)
        for index, color in enumerate(colors):
            hue = (index * plot_module.GOLDEN_ANGLE_DEG) % 360.0
            expected = colorsys.hsv_to_rgb(hue / 360.0, 0.75, 0.9)
            assert color == pytest.approx(expected)


class TestPlotBeforeAfter:
    def test_calls_show_once(self, shown_figures):
        plot_module.plot_before_after(np.asarray(POINTS, dtype=float), make_miner(0.5))
        assert len(shown_figures) == 1

    def test_figure_size_and_axes_titles(self, shown_figures):
        plot_module.plot_before_after(np.asarray(POINTS, dtype=float), make_miner(0.5))
        fig = shown_figures[0]

        assert len(fig.axes) == 2
        assert tuple(np.round(fig.get_size_inches(), 6)) == (14.0, 6.0)
        assert fig.get_suptitle() == (
            "Quality Threshold Clustering (PCA 2D projection)"
        )

        before, after = fig.axes
        assert before.get_title() == "Before Clustering"
        assert after.get_title() == "After Clustering"
        for axis in (before, after):
            assert axis.get_xlabel() == "PC1"
            assert axis.get_ylabel() == "PC2"

    def test_before_panel_plots_every_sample_in_steelblue(self, shown_figures):
        X = np.asarray(POINTS, dtype=float)
        plot_module.plot_before_after(X, make_miner(0.5))
        before = shown_figures[0].axes[0]

        assert len(before.collections) == 1
        collection = before.collections[0]
        assert np.asarray(collection.get_offsets()).shape == (len(POINTS), 2)
        facecolors = np.asarray(collection.get_facecolors())
        assert facecolors.shape[0] >= 1
        assert np.allclose(facecolors[0, :3], STEELBLUE, atol=1e-6)

    def test_after_panel_matches_miner_clusters(self, shown_figures):
        X = np.asarray(POINTS, dtype=float)
        miner = make_miner(0.5)
        plot_module.plot_before_after(X, miner)

        after = shown_figures[0].axes[1]
        clusters = miner.clusters
        labels = miner.get_labels(X.shape[0])
        colors = plot_module.generate_colors(len(clusters))

        assert miner.cluster_count == 2
        assert np.all(labels >= 0)
        assert len(after.collections) == len(clusters)

        expected_labels = [
            f"Cluster {index + 1} ({len(cluster)} samples)"
            for index, cluster in enumerate(clusters)
        ]
        assert [
            text.get_text() for text in after.get_legend().get_texts()
        ] == expected_labels

        plotted = 0
        for index, cluster in enumerate(clusters):
            collection = after.collections[index]
            offsets = np.asarray(collection.get_offsets())
            assert offsets.shape == (len(cluster), 2)
            assert collection.get_label() == expected_labels[index]
            facecolors = np.asarray(collection.get_facecolors())
            assert facecolors.shape[0] >= 1
            assert np.allclose(facecolors[0, :3], colors[index], atol=1e-6)
            plotted += len(cluster)

        assert plotted == X.shape[0]

    def test_single_cluster(self, shown_figures):
        X = np.asarray(POINTS, dtype=float)
        miner = make_miner(10.0)
        plot_module.plot_before_after(X, miner)
        after = shown_figures[0].axes[1]

        assert miner.cluster_count == 1
        assert len(after.collections) == 1
        assert np.asarray(after.collections[0].get_offsets()).shape == (
            len(POINTS),
            2,
        )
        assert [text.get_text() for text in after.get_legend().get_texts()] == [
            f"Cluster 1 ({len(POINTS)} samples)"
        ]

    def test_singleton_clusters(self, shown_figures):
        X = np.asarray(POINTS, dtype=float)
        miner = make_miner(0.001)
        plot_module.plot_before_after(X, miner)
        after = shown_figures[0].axes[1]

        assert miner.cluster_count == len(POINTS)
        assert miner.outlier_percentage() == 100.0
        assert len(after.collections) == len(POINTS)
        assert all(
            np.asarray(collection.get_offsets()).shape == (1, 2)
            for collection in after.collections
        )

    def test_points_use_pca_projection(self, shown_figures):
        from sklearn.decomposition import PCA

        X = np.asarray(POINTS, dtype=float)
        plot_module.plot_before_after(X, make_miner(0.5))
        before = shown_figures[0].axes[0]

        projected = PCA(n_components=2).fit_transform(X)
        offsets = np.asarray(before.collections[0].get_offsets())
        assert np.allclose(offsets, projected)
