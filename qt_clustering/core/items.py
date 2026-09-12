from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .attributes import Attribute, ContinuousAttribute


@dataclass(frozen=True)
class Item(ABC):
    attribute: Attribute
    value: object

    @abstractmethod
    def distance(self, other_value: object) -> float: ...


@dataclass(frozen=True)
class ContinuousItem(Item):
    value: float = 0.0

    def distance(self, other_value: object) -> float:
        if not isinstance(other_value, (int, float)):
            raise TypeError(
                f"ContinuousItem.distance() expects numeric value, got {type(other_value).__name__}"
            )
        attr: ContinuousAttribute = self.attribute  # type: ignore[assignment]
        scaled_self = attr.get_scaled_value(self.value)
        scaled_other = attr.get_scaled_value(float(other_value))
        return abs(scaled_self - scaled_other)


@dataclass(frozen=True)
class DiscreteItem(Item):
    value: str = ""

    def distance(self, other_value: object) -> float:
        return 0.0 if self.value == other_value else 1.0
