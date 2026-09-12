from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .items import Item

if TYPE_CHECKING:
    from .data import Data


@dataclass
class QTuple:
    items: list[Item] = field(default_factory=list)

    def get_length(self) -> int:
        return len(self.items)

    def get(self, index: int) -> Item:
        return self.items[index]

    def get_distance(self, other: QTuple) -> float:
        if self.get_length() != other.get_length():
            raise ValueError(
                f"Cannot compute distance between tuples of different lengths: "
                f"{self.get_length()} vs {other.get_length()}"
            )
        return sum(
            item.distance(other.items[i].value)
            for i, item in enumerate(self.items)
        )

    def avg_distance(self, data: "Data", clustered_ids: set[int]) -> float:
        if not clustered_ids:
            return 0.0
        total = 0.0
        for idx in clustered_ids:
            total += self.get_distance(data.get_item_set(idx))
        return total / len(clustered_ids)
