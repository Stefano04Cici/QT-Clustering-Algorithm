from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qt_clustering.core.data import Data
from qt_clustering.menu import RECOMMENDED_RADIUS, load_dataset, prompt_radius, run


# --- prompt_radius tests ---


@patch("builtins.input", return_value="2.0")
def test_prompt_radius_valid(mock_input):
    assert prompt_radius() == 2.0


@patch("builtins.input", return_value="")
def test_prompt_radius_default(mock_input):
    assert prompt_radius() == RECOMMENDED_RADIUS


@patch("builtins.input", side_effect=["abc", "1.5"])
def test_prompt_radius_invalid_then_valid(mock_input):
    assert prompt_radius() == 1.5


@patch("builtins.input", side_effect=["-1", "0", "1.5"])
def test_prompt_radius_negative_then_valid(mock_input):
    assert prompt_radius() == 1.5


@patch("builtins.input", return_value="0")
def test_prompt_radius_zero_rejected(mock_input):
    with patch("builtins.input", return_value="1.5"):
        result = prompt_radius()
        assert result == 1.5


# --- load_dataset tests ---


@patch("qt_clustering.menu.make_blobs")
def test_load_dataset(mock_fetch):
    mock_x = np.zeros((10, 2))
    mock_y = np.zeros(10, dtype=int)
    mock_fetch.return_value = (mock_x, mock_y)

    data, X, names, y_true = load_dataset()
    assert data.get_number_of_examples() == 10
    assert data.get_number_of_attributes() == 2
    assert names == ["feature_0", "feature_1"]
    assert len(y_true) == 10


# --- run tests ---


def _make_blobs(n_samples: int = 20, n_features: int = 2):
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=0.0, scale=0.1, size=(n_samples // 2, n_features))
    c2 = rng.normal(loc=5.0, scale=0.1, size=(n_samples // 2, n_features))
    X = np.vstack([c1, c2])
    y = np.zeros(n_samples, dtype=int)
    return X, y


@patch("qt_clustering.menu.make_blobs")
@patch("builtins.input", side_effect=["1.0", "n", "n"])
def test_run_normal_flow(mock_input, mock_fetch):
    mock_fetch.return_value = _make_blobs()
    run()


@patch("qt_clustering.menu.make_blobs")
@patch("builtins.input", side_effect=["100.0", "1.0", "n", "n"])
def test_run_radius_too_large_then_retry(mock_input, mock_fetch):
    mock_fetch.return_value = _make_blobs()
    run()


@patch("qt_clustering.menu.make_blobs")
@patch("builtins.input", side_effect=["", "1.0", "n", "n"])
def test_run_default_radius(mock_input, mock_fetch):
    mock_fetch.return_value = _make_blobs()
    run()


@patch("qt_clustering.menu.make_blobs")
@patch("qt_clustering.plot.plt")
@patch("builtins.input", side_effect=["1.0", "y", "n", "n"])
def test_run_retry_loop(mock_input, mock_plt, mock_fetch):
    mock_fetch.return_value = _make_blobs()
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
    run()


@patch("qt_clustering.menu.make_blobs")
def test_clustering_metrics(mock_fetch):
    mock_fetch.return_value = _make_blobs()

    data, X, names, y_true = load_dataset()

    from qt_clustering.core.qt_miner import QTMiner
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    miner = QTMiner(radius=0.5)
    miner.compute(data)
    y_pred = miner.cluster_set.get_labels(len(y_true))

    ari = adjusted_rand_score(y_true, y_pred)
    sil = silhouette_score(X, y_pred)
    outlier_pct = miner.cluster_set.outlier_percentage()

    assert 0.0 <= ari <= 1.0
    assert -1.0 <= sil <= 1.0
    assert 0.0 <= outlier_pct <= 100.0
