from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import Cluster


@dataclass
class ClusterSet:
    clusters: list[Cluster] = field(default_factory=list)

    def add(self, cluster: Cluster) -> None:
        self.clusters.append(cluster)
        self.clusters.sort()

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, index: int) -> Cluster:
        return self.clusters[index]

    def get_number_of_clusters(self) -> int:
        return len(self.clusters)

    def get_labels(self, n_samples: int) -> np.ndarray:
        labels = np.full(n_samples, -1, dtype=int)
        for cluster_idx, cluster in enumerate(self):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def outlier_percentage(self) -> float:
        total = sum(c.get_size() for c in self.clusters)
        if total == 0:
            return 0.0
        outliers = sum(1 for c in self.clusters if c.get_size() == 1)
        return (outliers / total) * 100

    def __str__(self) -> str:
        lines = [f"ClusterSet({self.get_number_of_clusters()} clusters):"]
        for i, c in enumerate(self.clusters):
            lines.append(f"  {i + 1}. {c}")
        return "\n".join(lines)
