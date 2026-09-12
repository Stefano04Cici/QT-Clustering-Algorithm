from __future__ import annotations

from .cluster import Cluster
from .cluster_set import ClusterSet
from .data import Data
from qt_clustering.exceptions import ClusteringRadiusException, EmptyDatasetException
from .tuple import QTuple


class QTMiner:
    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("radius must be > 0")
        self.radius = radius
        self.cluster_set = ClusterSet()

    def compute(self, data: Data) -> int:
        if data.get_number_of_examples() == 0:
            raise EmptyDatasetException("Dataset is empty")

        n = data.get_number_of_examples()
        # Cache all item sets to avoid repeated creation
        item_sets: list[QTuple] = [data.get_item_set(i) for i in range(n)]
        is_clustered = [False] * n

        while sum(is_clustered) < n:
            candidate = self._build_candidate_cluster(item_sets, is_clustered)
            self.cluster_set.add(candidate)
            for idx in candidate:
                is_clustered[idx] = True

        return self.cluster_set.get_number_of_clusters()

    def _build_candidate_cluster(
        self, item_sets: list[QTuple], is_clustered: list[bool]
    ) -> Cluster:
        n = len(item_sets)
        best_cluster: Cluster | None = None

        for i in range(n):
            if is_clustered[i]:
                continue

            centroid = item_sets[i]
            candidate = Cluster(centroid=centroid)

            for j in range(n):
                if is_clustered[j] or i == j:
                    continue
                if centroid.get_distance(item_sets[j]) <= self.radius:
                    candidate.add_data(j)

            candidate.add_data(i)

            if best_cluster is None or candidate.get_size() > best_cluster.get_size():
                best_cluster = candidate

        return best_cluster  # type: ignore[return-value]
