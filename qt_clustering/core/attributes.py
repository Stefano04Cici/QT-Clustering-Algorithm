from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute(ABC):
    name: str
    index: int


@dataclass(frozen=True)
class ContinuousAttribute(Attribute):
    min_val: float = 0.0
    max_val: float = 0.0

    def get_scaled_value(self, v: float) -> float:
        r = self.max_val - self.min_val
        if r == 0:
            return 0.0
        return (v - self.min_val) / r


@dataclass(frozen=True)
class DiscreteAttribute(Attribute):
    values: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return len(self.values)
