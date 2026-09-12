import os
import tempfile

import numpy as np
import pytest

from qt_clustering.core.attributes import ContinuousAttribute

from qt_clustering.core.data import Data


class TestFromNumpy:
    def test_basic(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data = Data.from_numpy(X, ["a", "b"])
        assert data.get_number_of_examples() == 3
        assert data.get_number_of_attributes() == 2
        assert data.attributes[0].name == "a"
        assert data.attributes[1].name == "b"

    def test_values(self):
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        data = Data.from_numpy(X, ["x", "y"])
        assert data.get_value(0, 0) == 10.0
        assert data.get_value(1, 1) == 40.0

    def test_min_max(self):
        X = np.array([[1.0], [5.0], [3.0]])
        data = Data.from_numpy(X, ["val"])
        attr = data.attributes[0]
        assert isinstance(attr, ContinuousAttribute)
        assert attr.min_val == 1.0
        assert attr.max_val == 5.0

    def test_default_names(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = Data.from_numpy(X)
        assert data.attributes[0].name == "feat_0"
        assert data.attributes[1].name == "feat_1"

    def test_get_item_set(self):
        X = np.array([[1.0, 2.0]])
        data = Data.from_numpy(X, ["a", "b"])
        t = data.get_item_set(0)
        assert t.get_length() == 2
        assert t.get(0).value == 1.0


class TestFromCSV:
    def test_basic(self):
        content = "x,y\n1.0,2.0\n3.0,4.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            data = Data.from_csv(path)
            assert data.get_number_of_examples() == 2
            assert data.get_number_of_attributes() == 2
            assert data.attributes[0].name == "x"
        finally:
            os.unlink(path)

    def test_mixed_types(self):
        content = "name,value\nhello,1.0\nworld,2.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            data = Data.from_csv(path)
            assert data.get_number_of_attributes() == 2
        finally:
            os.unlink(path)

    def test_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            data = Data.from_csv(path)
            assert data.get_number_of_examples() == 0
        finally:
            os.unlink(path)
