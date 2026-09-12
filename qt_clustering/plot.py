from __future__ import annotations

import colorsys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from qt_clustering.core.cluster_set import ClusterSet

GOLDEN_ANGLE_DEG = 137.508


def generate_colors(n: int) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        hue = (i * GOLDEN_ANGLE_DEG) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.75, 0.9)
        colors.append((r, g, b))
    return colors


def plot_before_after(
    X: np.ndarray,
    cluster_set: ClusterSet,
) -> None:
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c="steelblue", s=15, alpha=0.7, edgecolors="none")
    ax1.set_title("Before Clustering")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.3)

    labels = np.full(X.shape[0], -1, dtype=int)
    for cluster_idx, cluster in enumerate(cluster_set):
        for sample_idx in cluster:
            labels[sample_idx] = cluster_idx

    colors = generate_colors(cluster_set.get_number_of_clusters())

    for cluster_idx, cluster in enumerate(cluster_set):
        mask = labels == cluster_idx
        ax2.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors[cluster_idx]],
            s=15,
            alpha=0.7,
            edgecolors="none",
            label=f"Cluster {cluster_idx + 1} ({cluster.get_size()} samples)",
        )

    ax2.set_title("After Clustering")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Quality Threshold Clustering (PCA 2D projection)", fontsize=13)
    plt.tight_layout()
    plt.show()
