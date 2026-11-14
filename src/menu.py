from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.datasets import make_blobs

from src.core.qt_miner import QTMiner

RECOMMENDED_RADIUS = 0.50


def load_dataset() -> tuple[np.ndarray, list[str], np.ndarray]:
    blobs = make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        center_box=(-10.0, 10.0),
        random_state=42,
        return_centers=False,
    )
    X, y = blobs[0], blobs[1]
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return X, feature_names, y


def prompt_radius() -> float:
    while True:
        raw = input(
            f"Enter clustering radius (recommended: {RECOMMENDED_RADIUS}): "
        ).strip()
        if raw == "":
            print(f"Using default radius: {RECOMMENDED_RADIUS}")
            return RECOMMENDED_RADIUS
        try:
            radius = float(raw)
            if radius <= 0:
                print("Radius must be greater than 0. Try again.")
                continue
            return radius
        except ValueError:
            print("Invalid number. Try again.")


def dataset_preview(
    X: np.ndarray, feature_names: list[str], limit: int = 10
) -> str:
    header = " | ".join(feature_names)
    lines = [header, "-" * len(header)]
    for row in X[:limit]:
        lines.append(" | ".join(str(float(value)) for value in row))
    if X.shape[0] > limit:
        lines.append(f"... ({X.shape[0]} rows total)")
    return "\n".join(lines)


def cluster_details(miner: QTMiner) -> list[str]:
    return [
        line.split(". ", 1)[-1]
        for line in str(miner).splitlines()[1:]
    ]


def display_results(
    miner: QTMiner,
    y_true: np.ndarray | None = None,
    X: np.ndarray | None = None,
) -> None:
    n_clusters = miner.cluster_count
    details = cluster_details(miner)

    print(f"\nNumber of clusters: {n_clusters}\n")
    for i, cluster in enumerate(miner.clusters):
        print(f"  Cluster {i + 1}: {len(cluster)} samples")
        print(f"    Centroid: {details[i]}")
    print()

    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score

        y_pred = miner.get_labels(len(y_true))
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"\nClustering evaluation:\n  \n  ARI: {ari:.4f}")

    if X is not None and n_clusters > 1:
        from sklearn.metrics import silhouette_score

        y_pred = miner.get_labels(X.shape[0])
        sil = silhouette_score(X, y_pred)
        print(f"  Silhouette Coefficient: {sil:.4f}")

    print(f"  Outlier Percentage: {miner.outlier_percentage():.4f}")
    print(f"  \nTotal clustering time: {miner.elapsed_ms:.2f} ms\n")


def run() -> None:
    print("=" * 50)
    print("  Quality Threshold Clustering")
    print("=" * 50)
    print("\nLoading dataset...")

    X, feature_names, y_true = load_dataset()

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}\n")
    print(dataset_preview(X, feature_names))
    print()

    while True:
        radius = prompt_radius()

        print(f"\nRunning QT clustering with radius={radius}...")
        miner = QTMiner(radius=radius)
        miner.compute(X, feature_names=feature_names)

        display_results(miner, y_true, X)

        show = input("Show clustering plot? (y/n): ").strip().lower()
        if show == "y":
            from src.plot import plot_before_after

            plot_before_after(X, miner)

        again = input("Run again with a different radius? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!")
            break
        print()


if __name__ == "__main__":
    run()
