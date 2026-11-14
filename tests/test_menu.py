from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import src.menu as menu
from src.core.qt_miner import QTMiner

POINTS = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
FEATURE_NAMES = ["x", "y"]
TRUE_LABELS = np.asarray([0, 0, 1, 1])
MENU_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "menu.py"


def feed_input(monkeypatch, values):
    remaining = iter(values)

    def fake_input(prompt=""):
        try:
            return next(remaining)
        except StopIteration:
            raise AssertionError(f"Unexpected extra input: {prompt!r}") from None

    monkeypatch.setattr("builtins.input", fake_input)


def make_miner(radius=0.5):
    miner = QTMiner(radius)
    miner.compute(POINTS, feature_names=FEATURE_NAMES)
    return miner


@pytest.fixture
def small_dataset(monkeypatch):
    X = np.asarray(POINTS, dtype=float)
    monkeypatch.setattr(
        menu, "load_dataset", lambda: (X, FEATURE_NAMES, TRUE_LABELS)
    )
    return X


@pytest.fixture
def plot_calls(monkeypatch):
    calls = []
    module = types.ModuleType("src.plot")
    module.plot_before_after = lambda X, miner: calls.append((X, miner))
    monkeypatch.setitem(sys.modules, "src.plot", module)
    return calls


class TestLoadDataset:
    def test_shape_features_and_labels(self):
        X, feature_names, y = menu.load_dataset()

        assert X.shape == (300, 2)
        assert feature_names == ["feature_0", "feature_1"]
        assert y.shape == (300,)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_is_deterministic(self):
        X1, _, y1 = menu.load_dataset()
        X2, _, y2 = menu.load_dataset()

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestPromptRadius:
    def test_empty_input_uses_recommended_radius(self, monkeypatch, capsys):
        feed_input(monkeypatch, [""])

        assert menu.prompt_radius() == menu.RECOMMENDED_RADIUS
        assert (
            f"Using default radius: {menu.RECOMMENDED_RADIUS}"
            in capsys.readouterr().out
        )

    def test_accepts_float(self, monkeypatch):
        feed_input(monkeypatch, ["0.75"])

        assert menu.prompt_radius() == pytest.approx(0.75)

    def test_strips_surrounding_whitespace(self, monkeypatch):
        feed_input(monkeypatch, ["  0.25  "])

        assert menu.prompt_radius() == pytest.approx(0.25)

    def test_accepts_scientific_notation(self, monkeypatch):
        feed_input(monkeypatch, ["1e-2"])

        assert menu.prompt_radius() == pytest.approx(0.01)

    def test_retries_after_invalid_number(self, monkeypatch, capsys):
        feed_input(monkeypatch, ["abc", "0.5"])

        assert menu.prompt_radius() == pytest.approx(0.5)
        assert "Invalid number. Try again." in capsys.readouterr().out

    def test_retries_after_non_positive_radius(self, monkeypatch, capsys):
        feed_input(monkeypatch, ["0", "-2", "0.5"])

        assert menu.prompt_radius() == pytest.approx(0.5)
        output = capsys.readouterr().out
        assert output.count("Radius must be greater than 0. Try again.") == 2


class TestDatasetPreview:
    def test_header_rows_and_no_suffix_when_all_rows_fit(self):
        X = np.arange(20, dtype=float).reshape(10, 2)

        lines = menu.dataset_preview(X, ["a", "b"]).splitlines()

        assert lines[0] == "a | b"
        assert lines[1] == "-----"
        assert lines[2] == "0.0 | 1.0"
        assert lines[11] == "18.0 | 19.0"
        assert len(lines) == 12
        assert "(10 rows total)" not in "\n".join(lines)

    def test_reports_total_when_rows_exceed_limit(self):
        X = np.arange(24, dtype=float).reshape(12, 2)

        lines = menu.dataset_preview(X, ["a", "b"]).splitlines()

        assert len(lines) == 2 + 10 + 1
        assert lines[-1] == "... (12 rows total)"

    def test_custom_limit(self):
        X = np.arange(20, dtype=float).reshape(10, 2)

        lines = menu.dataset_preview(X, ["a", "b"], limit=3).splitlines()

        assert len(lines) == 2 + 3 + 1
        assert lines[-1] == "... (10 rows total)"

    def test_empty_dataset_only_contains_header_and_separator(self):
        X = np.empty((0, 2), dtype=float)

        lines = menu.dataset_preview(X, ["a", "b"]).splitlines()

        assert lines == ["a | b", "-----"]


class TestClusterDetails:
    def test_one_detail_per_cluster(self):
        miner = make_miner(0.5)

        details = menu.cluster_details(miner)

        assert len(details) == miner.cluster_count == 2
        assert all(detail.startswith("Cluster(size=") for detail in details)

    def test_matches_string_representation_without_numbering(self):
        miner = make_miner(0.5)

        expected = [
            line.split(". ", 1)[-1]
            for line in str(miner).splitlines()[1:]
        ]

        assert menu.cluster_details(miner) == expected
        assert all("centroid={" in detail for detail in expected)
        assert all("x=" in detail and "y=" in detail for detail in expected)


class TestDisplayResults:
    def test_prints_cluster_summary_and_timing(self, capsys):
        miner = make_miner(0.5)

        menu.display_results(miner)
        output = capsys.readouterr().out

        assert "Number of clusters: 2" in output
        assert "  Cluster 1: 2 samples" in output
        assert "  Cluster 2: 2 samples" in output
        for detail in menu.cluster_details(miner):
            assert f"    Centroid: {detail}" in output
        assert "Outlier Percentage: 0.0000" in output
        assert "Total clustering time:" in output
        assert "ARI" not in output
        assert "Silhouette" not in output

    def test_prints_elapsed_time_with_two_decimals(self, capsys):
        menu.display_results(make_miner(0.5))
        output = capsys.readouterr().out

        prefix = "Total clustering time: "
        line = next(
            line for line in output.splitlines() if line.startswith(prefix)
        )
        value = line[len(prefix) :].removesuffix(" ms")

        assert float(value) >= 0.0
        assert len(value.split(".")[-1]) == 2

    def test_prints_ari_when_true_labels_are_available(self, capsys):
        menu.display_results(make_miner(0.5), y_true=TRUE_LABELS)
        output = capsys.readouterr().out

        assert "Clustering evaluation" in output
        assert "ARI: 1.0000" in output
        assert "Silhouette" not in output

    def test_prints_silhouette_when_data_has_multiple_clusters(self, capsys):
        menu.display_results(
            make_miner(0.5), y_true=TRUE_LABELS, X=np.asarray(POINTS)
        )
        output = capsys.readouterr().out

        assert "ARI: 1.0000" in output
        assert "Silhouette Coefficient: 0.9900" in output

    def test_skips_silhouette_for_single_cluster(self, capsys):
        menu.display_results(
            make_miner(10.0), y_true=TRUE_LABELS, X=np.asarray(POINTS)
        )
        output = capsys.readouterr().out

        assert "Number of clusters: 1" in output
        assert "ARI: 0.0000" in output
        assert "Silhouette" not in output

    def test_reports_all_singletons_as_outliers(self, capsys):
        menu.display_results(make_miner(0.001))
        output = capsys.readouterr().out

        assert "Number of clusters: 4" in output
        assert "Outlier Percentage: 100.0000" in output


class TestRun:
    def test_single_iteration_without_plot(
        self, small_dataset, plot_calls, monkeypatch, capsys
    ):
        feed_input(monkeypatch, ["0.5", "n", "n"])

        menu.run()
        output = capsys.readouterr().out

        assert "Quality Threshold Clustering" in output
        assert "Dataset loaded: 4 samples, 2 features" in output
        assert "Features: ['x', 'y']" in output
        assert "x | y" in output
        assert "Running QT clustering with radius=0.5" in output
        assert "Number of clusters: 2" in output
        assert "ARI: 1.0000" in output
        assert "Silhouette Coefficient: 0.9900" in output
        assert "Total clustering time:" in output
        assert "Goodbye!" in output
        assert plot_calls == []

    def test_plots_when_requested(
        self, small_dataset, plot_calls, monkeypatch, capsys
    ):
        feed_input(monkeypatch, ["0.5", "y", "n"])

        menu.run()

        assert len(plot_calls) == 1
        X, miner = plot_calls[0]
        np.testing.assert_array_equal(X, small_dataset)
        assert isinstance(miner, QTMiner)
        assert miner.radius == pytest.approx(0.5)
        assert miner.cluster_count == 2
        assert "Goodbye!" in capsys.readouterr().out

    def test_repeats_until_user_stops(
        self, small_dataset, plot_calls, monkeypatch, capsys
    ):
        feed_input(monkeypatch, ["0.5", "n", "y", "1.0", "n", "n"])

        menu.run()
        output = capsys.readouterr().out

        assert output.count("Running QT clustering with radius=") == 2
        assert "Running QT clustering with radius=0.5" in output
        assert "Running QT clustering with radius=1.0" in output
        assert output.count("Total clustering time:") == 2
        assert output.count("Goodbye!") == 1
        assert plot_calls == []

    def test_empty_radius_uses_default(
        self, small_dataset, plot_calls, monkeypatch, capsys
    ):
        feed_input(monkeypatch, ["", "n", "n"])

        menu.run()
        output = capsys.readouterr().out

        assert f"Using default radius: {menu.RECOMMENDED_RADIUS}" in output
        assert f"radius={menu.RECOMMENDED_RADIUS}" in output

    def test_rejects_invalid_radii_before_running(
        self, small_dataset, plot_calls, monkeypatch, capsys
    ):
        feed_input(monkeypatch, ["abc", "0", "-2", "0.5", "n", "n"])

        menu.run()
        output = capsys.readouterr().out

        assert output.count("Invalid number. Try again.") == 1
        assert output.count("Radius must be greater than 0. Try again.") == 2
        assert output.count("Running QT clustering with radius=") == 1

    def test_script_entry_point_runs_menu(self, plot_calls, monkeypatch, capsys):
        feed_input(monkeypatch, ["0.5", "n", "n"])

        runpy.run_path(str(MENU_SCRIPT), run_name="__main__")
        output = capsys.readouterr().out

        assert "Quality Threshold Clustering" in output
        assert "Dataset loaded: 300 samples, 2 features" in output
        assert "Total clustering time:" in output
        assert "Goodbye!" in output
        assert plot_calls == []
