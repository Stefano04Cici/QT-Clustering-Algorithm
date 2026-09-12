from __future__ import annotations

from dataclasses import dataclass, field
from .data import Data

from .tuple import QTuple


@dataclass
class Cluster:
    centroid: QTuple
    clustered_data: set[int] = field(default_factory=set)

    def add_data(self, id: int) -> bool:
        if id in self.clustered_data:
            return False
        self.clustered_data.add(id)
        return True

    def contains(self, id: int) -> bool:
        return id in self.clustered_data

    def get_size(self) -> int:
        return len(self.clustered_data)

    def __iter__(self):
        return iter(self.clustered_data)

    def __lt__(self, other: Cluster) -> bool:
        if self.get_size() != other.get_size():
            return self.get_size() < other.get_size()
        for i in range(min(self.centroid.get_length(), other.centroid.get_length())):
            v1 = self.centroid.get(i).value
            v2 = other.centroid.get(i).value
            if v1 != v2:
                # Compare numerically if both values are numeric-like, otherwise string comparison
                if isinstance(v1, (int, float, str)) and isinstance(v2, (int, float, str)):
                    try:
                        return float(v1) < float(v2)
                    except (ValueError, TypeError):
                        return str(v1) < str(v2)
                return str(v1) < str(v2)
        return self.centroid.get_length() < other.centroid.get_length()

    def __str__(self) -> str:
        centroid_vals = ", ".join(
            f"{self.centroid.get(i).attribute.name}={self.centroid.get(i).value}"
            for i in range(self.centroid.get_length())
        )
        return f"Cluster(size={self.get_size()}, centroid={{{centroid_vals}}})"

    def to_string(self, data: Data) -> str:
        lines = [str(self)]
        for idx in self.clustered_data:
            example = data.examples[idx]
            dist = self.centroid.get_distance(data.get_item_set(idx))
            vals = ", ".join(
                f"{data.attributes[i].name}={example[i]}"
                for i in range(len(data.attributes))
            )
            lines.append(f"  Example {idx}: {{{vals}}} distance={dist:.4f}")
        avg = self.centroid.avg_distance(data, self.clustered_data)
        lines.append(f"  Average distance: {avg:.4f}")
        return "\n".join(lines)
