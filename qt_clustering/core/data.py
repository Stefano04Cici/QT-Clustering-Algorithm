from __future__ import annotations

import csv
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .attributes import (
    Attribute,
    ContinuousAttribute,
    DiscreteAttribute,
)

from .items import ContinuousItem, DiscreteItem, Item

from .tuple import QTuple


@dataclass
class Data:
    examples: list[list[object]] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)

    @classmethod
    def from_csv(cls, filepath: str) -> Data:
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return cls()

        header = rows[0]
        data_rows = rows[1:]
        ncols = len(header)
        attributes: list[Attribute] = []

        for col_idx in range(ncols):
            col_values = [row[col_idx] for row in data_rows if row[col_idx] != ""]
            numeric_vals = []
            all_numeric = True
            for v in col_values:
                try:
                    numeric_vals.append(float(v))
                except ValueError:
                    all_numeric = False
                    break

            if all_numeric and numeric_vals:
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                attributes.append(
                    ContinuousAttribute(
                        name=header[col_idx], index=col_idx, min_val=min_val, max_val=max_val
                    )
                )
            else:
                distinct = frozenset(col_values)
                attributes.append(
                    DiscreteAttribute(name=header[col_idx], index=col_idx, values=distinct)
                )

        examples: list[list[object]] = [list(row) for row in data_rows]
        return cls(examples=examples, attributes=attributes)

    @classmethod
    def from_numpy(cls, X: np.ndarray, feature_names: list[str] | None = None) -> Data:
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return cls(examples=[], attributes=[])

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_cols)]

        attributes: list[Attribute] = []
        for col_idx in range(n_cols):
            col = X[:, col_idx].astype(float)
            min_val = float(np.min(col))
            max_val = float(np.max(col))
            attributes.append(
                ContinuousAttribute(
                    name=feature_names[col_idx],
                    index=col_idx,
                    min_val=min_val,
                    max_val=max_val,
                )
            )

        examples: list[list[object]] = [
            [float(X[r, c]) for c in range(n_cols)] for r in range(n_rows)
        ]
        return cls(examples=examples, attributes=attributes)

    def get_number_of_examples(self) -> int:
        return len(self.examples)

    def get_number_of_attributes(self) -> int:
        return len(self.attributes)

    def get_attribute(self, index: int) -> Attribute:
        if not 0 <= index < len(self.attributes):
            raise IndexError(f"index {index} out of range [0, {len(self.attributes)})")
        return self.attributes[index]

    def get_value(self, example_idx: int, attribute_idx: int) -> object:
        if not 0 <= example_idx < len(self.examples):
            raise IndexError(f"example_idx {example_idx} out of range [0, {len(self.examples)})")
        if not 0 <= attribute_idx < len(self.attributes):
            raise IndexError(f"attribute_idx {attribute_idx} out of range [0, {len(self.attributes)})")
        return self.examples[example_idx][attribute_idx]

    def get_item_set(self, index: int) -> QTuple:
        if not 0 <= index < len(self.examples):
            raise IndexError(f"index {index} out of range [0, {len(self.examples)})")
        items: list[Item] = []
        for attr in self.attributes:
            raw = self.examples[index][attr.index]
            if isinstance(attr, ContinuousAttribute):
                value = float(raw) if isinstance(raw, Real) else float(str(raw))
                items.append(ContinuousItem(attribute=attr, value=value))
            elif isinstance(attr, DiscreteAttribute):
                items.append(DiscreteItem(attribute=attr, value=str(raw)))
        return QTuple(items=items)

    def __str__(self) -> str:
        lines = []
        header = " | ".join(a.name for a in self.attributes)
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.examples[:10]:
            lines.append(" | ".join(str(v) for v in row))
        if len(self.examples) > 10:
            lines.append(f"... ({self.get_number_of_examples()} rows total)")
        return "\n".join(lines)
