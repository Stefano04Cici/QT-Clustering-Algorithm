# Core Module Description

This document describes the role of every class defined in the `src/core` folder.

The core is implemented in C++17 inside the `qt` namespace. Each module lives in its own `.cpp` file, has an include guard, and includes the previous one, so the whole core is compiled as a single translation unit in a fixed order:

`attributes.cpp` → `items.cpp` → `tuple.cpp` → `data.cpp` → `cluster.cpp` → `cluster_set.cpp` → `qt_miner.cpp`

`bindings.cpp` includes `qt_miner.cpp` and turns the core into the `_qt_miner` Python extension. Only `QTMiner` is exposed to Python; every other class is internal.

## attributes.cpp

Defines the feature metadata shared by all items and datasets.

### Attribute

Abstract base class for a dataset column.

- Stores the column `name_` (`std::string`) and `index_` (`int`).
- `name()` and `index()` return the two metadata fields.
- Pure virtual `is_continuous()` and `equals(const Attribute&)`.
- `num_values()` returns `0` by default and is overridden only by discrete attributes.
- `get_scaled_value(double)` has no meaning in the base class and throws `std::logic_error`; it is overridden by `ContinuousAttribute`.
- Supports copy/move construction and assignment; the destructor is virtual.
- `operator==`/`operator!=` delegate to `equals`.

### ContinuousAttribute

Final class describing a numeric feature.

- Stores `min_val_` and `max_val_`, the bounds used for normalization.
- `get_scaled_value(v)` returns `(v - min_val_) / (max_val_ - min_val_)`. When the range is zero it returns `0.0`, avoiding a division by zero.
- `is_continuous()` returns `true`.
- `equals` checks the dynamic type, name, index, minimum and maximum.

### DiscreteAttribute

Final class describing a categorical feature.

- `ValueSet` is `std::set<std::string>`, so the admissible values are unique and ordered.
- Stores the set of known values; `values()` exposes it and `num_values()` returns its size.
- `is_continuous()` returns `false`.
- `equals` checks the dynamic type, name, index and value set.

## items.cpp

Defines the value of a single feature of a single example.

### Item

Abstract base class for one feature value.

- `Value` is `std::variant<double, std::string>`: numeric values for continuous features, strings for discrete ones.
- Stores a non-owning `const Attribute*` plus the `Value`. The attribute must outlive the item (see the lifetime notes on `Data` and the binding).
- `attribute()` returns the referenced attribute; `value()` returns the variant.
- Pure virtual `is_continuous()`, `distance(const Value&)` and `equals(const Item&)`.
- `operator==`/`operator!=` delegate to `equals`.

### ContinuousItem

Final item for a numeric feature.

- Constructed from a `ContinuousAttribute` and a `double`; `value()` returns the number.
- `attribute()` narrows the base reference to `const ContinuousAttribute&`.
- `distance(other)` computes `|scaled(value) - scaled(other)|`; the result is a per-feature contribution between 0 and 1. It throws `std::invalid_argument` if the other variant does not hold a `double`.
- `equals` requires the same attribute and the same numeric value.

### DiscreteItem

Final item for a categorical feature.

- Constructed from a `DiscreteAttribute` and a `std::string`; `value()` returns the string.
- `attribute()` narrows the base reference to `const DiscreteAttribute&`.
- `distance(other)` returns `0.0` when the other value is the same string, and `1.0` otherwise (including when the variant does not hold a string).
- `equals` requires the same attribute and the same string value.

## tuple.cpp

### QTuple

Represents a complete example as a sequence of items.

- `ItemHolder` is `std::variant<ContinuousItem, DiscreteItem>`, so a tuple stores concrete items without heap allocation.
- Can be default-constructed, built from a `std::vector<ItemHolder>`, or built from an initializer list; `add()` appends one more item.
- `get_length()` returns the number of items; `items()` exposes the internal vector.
- `get(index)`/`operator[]` return an `Item&` or `const Item&` by visiting the variant; out-of-range indices throw through `std::vector::at`.
- `get_distance(other)` sums the per-item distances (Manhattan distance over normalized features). Tuples of different lengths are rejected with `std::invalid_argument`.
- `avg_distance(data, clustered_ids)` is a template helper used by `Cluster`: it averages the distance between this tuple (the centroid) and the examples identified by `clustered_ids` in `data`, returning `0.0` for an empty container.
- `equals` compares length and every item.

## data.cpp

### Data

Owns a dataset and its feature definitions.

- `Cell` is `Item::Value` (`std::variant<double, std::string>`); `AttributePtr` is `std::shared_ptr<const Attribute>`.
- The default constructor creates an empty dataset; the main constructor accepts the example matrix and the attribute vector.
- `from_csv(filepath)` reads a CSV file, skipping empty lines and stripping carriage returns. It treats the first row as the header and, for each column, tries to parse every non-empty value as a complete `double`: if all values are numeric the column becomes a `ContinuousAttribute` with the observed minimum and maximum, otherwise it becomes a `DiscreteAttribute` with the distinct strings. CSV rows keep the raw strings; numeric conversion happens later in `get_item_set`. Quoted fields are supported through a small internal CSV parser. A file without rows produces an empty `Data`.
- `from_matrix(matrix, feature_names)` builds a `Data` from a 2-D numeric matrix. Every column becomes a `ContinuousAttribute` with its minimum and maximum; missing names are generated as `feat_0`, `feat_1`, ... If fewer names than columns are supplied, `names.at` throws `std::out_of_range`. An empty matrix produces an empty `Data`.
- `examples()`/`attributes()` expose the underlying vectors; `get_number_of_examples()` and `get_number_of_attributes()` return their sizes.
- `get_attribute(index)` returns an attribute by reference and throws `std::out_of_range` when the index is invalid.
- `get_value(example_idx, attribute_idx)` returns a single raw cell, with bounds checks on both indices.
- `get_item_set(index)` converts one row into a `QTuple`: continuous attributes read the cell as a number (parsing the string for CSV data) and produce a `ContinuousItem`, discrete attributes produce a `DiscreteItem`.
- `to_string()` renders a table with the attribute names as header, the first 10 rows and a `... (N rows total)` suffix when needed; integral numeric values are rendered with a trailing `.0`.
- `equals` compares the examples and every attribute (type, name, index and bounds/values).
- Lifetime: the `Data` owns the attributes through `shared_ptr`, while items and cluster centroids only borrow `const Attribute*`. A `Data` must therefore outlive every `QTuple`/`Cluster` built from it; the Python binding enforces this.

## cluster.cpp

### Cluster

Groups a set of example indices around a representative example.

- `IdSet` is `std::set<int>`, so the clustered indices are unique and iterated in ascending order.
- Stores the centroid as a `QTuple` (a copy of the seed example) and the clustered indices.
- `centroid()` returns the centroid tuple (mutable and const overloads); `clustered_data()` exposes the index set.
- `add_data(id)` inserts an index and returns `true` only if it was not already present; `contains(id)` tests membership; `get_size()` returns the number of examples; `begin()`/`end()` allow iteration over the indices.
- `operator<` defines the canonical order used by `ClusterSet`: smaller clusters first, then lexicographic comparison of centroid values (numeric when both values parse as numbers, textual otherwise), then tuple length.
- `str()` returns `Cluster(size=N, centroid={name=value, ...})`.
- `to_string(data)` extends the description with every example (`Example <id>: {...} distance=<d>`) and the average distance from the centroid; distances are formatted with four decimals. The supplied `Data` must be the one the indices refer to.
- `equals` compares centroid and index set.

## cluster_set.cpp

### ClusterSet

Collection of all clusters produced by the algorithm.

- Stores `std::vector<Cluster> clusters_`; `add(cluster)` appends the cluster and re-sorts the vector with `Cluster::operator<`, so the collection is always ordered by size and centroid.
- Constructors accept no clusters or a pre-built vector; `clusters()` exposes the vector, `begin()`/`end()` allow iteration, and `size()`/`empty()`/`get_number_of_clusters()` report its size.
- `operator[]` returns a cluster by reference and throws `std::out_of_range` for an invalid index.
- `get_labels(n_samples)` returns a `std::vector<int>` of size `n_samples`, initialized to `-1`, where each clustered example receives the index of its cluster. If `n_samples` is smaller than a clustered index, `std::vector::at` throws `std::out_of_range`.
- `outlier_percentage()` counts singleton clusters over the total number of clustered examples and returns the percentage; an empty set returns `0.0`.
- `str()` returns `ClusterSet(N clusters):` followed by one numbered line per cluster.
- `equals` compares the cluster vectors.

## qt_miner.cpp

Defines the algorithm itself and its exception types.

### EmptyDatasetException

Subclass of `std::runtime_error`, thrown by `QTMiner::compute` when the dataset has no examples; pybind11 translates it into a Python `RuntimeError`.

### ClusteringRadiusException

Subclass of `std::runtime_error` reserved for radius-related clustering failures. Radius validation itself uses `std::invalid_argument`, which pybind11 translates into a Python `ValueError`.

### QTMiner

Implements the Quality Threshold algorithm.

- The constructor stores the radius and rejects values `<= 0` with `std::invalid_argument`; `set_radius` performs the same validation, while `radius()` returns the current value.
- `cluster_set()` exposes the internal `ClusterSet` with const and mutable overloads.
- `elapsed_ms()` returns the duration of the last clustering phase; the value is protected by a mutex because it may be read from another thread.
- `compute(data)`:
  1. throws `EmptyDatasetException` if the dataset has no examples;
  2. starts a `std::chrono::steady_clock` timer;
  3. converts every dataset row into a `QTuple`;
  4. repeatedly builds the best candidate cluster and adds it to the cluster set, marking its examples as clustered, until every example is assigned;
  5. stops the timer, stores the elapsed milliseconds under the mutex and returns the number of clusters.
- `cluster_set_` is not cleared at the start of `compute`: repeated calls append new clusters to the existing set, matching the original implementation.
- `build_candidate_cluster(...)` (private) scans the still-unclustered examples. For each one it builds a candidate containing the seed itself plus every unclustered example whose distance from the seed is `<= radius`, and keeps the largest candidate (the first one wins on ties). It throws `std::logic_error` if no candidate can be built.
- Members: `radius_`, `cluster_set_`, a `mutable std::mutex mutex_` guarding the elapsed time, and `elapsed_ms_` initialized to `0.0`.

## bindings.cpp

Defines the pybind11 module `_qt_miner`. Two helpers and one wrapper class are used.

### make_data (helper)

Converts a Python source into a `qt::Data`. A `str` or any object implementing `__fspath__` is treated as a CSV path; every other object is converted to a contiguous 2-D `double` NumPy array and passed to `Data::from_matrix` with the optional feature names. Non-2-D input raises a Python `ValueError`.

### labels_array (helper)

Copies a `std::vector<int>` into a one-dimensional NumPy `int32` array, which is what `labels`/`get_labels` return.

### PyQTMiner

Thin wrapper around `qt::QTMiner` and the only class registered in the Python module.

- Stores the C++ miner and a `std::vector<std::shared_ptr<qt::Data>>`: every dataset passed to `compute` is kept alive for the lifetime of the wrapper, because cluster centroids hold pointers to the attributes owned by that `Data`.
- `radius` is exposed as a read/write property backed by `QTMiner::radius`/`set_radius`; invalid values surface as Python `ValueError`.
- `compute(source, feature_names=None)` builds the `Data`, stores its `shared_ptr`, releases the GIL during `QTMiner::compute` and returns the number of clusters.
- `cluster_count`/`get_number_of_clusters()` return the number of clusters; `__len__` is bound to the same value.
- `clusters` returns a `list[list[int]]` with one ascending list of example indices per cluster.
- `labels(n_samples)` and its alias `get_labels(n_samples)` return the NumPy label array.
- `outlier_percentage()` returns the singleton-cluster percentage.
- `elapsed_ms` returns the duration of the last clustering phase.
- `__str__` returns the full `ClusterSet` description; `__repr__` returns `QTMiner(radius=..., clusters=...)`.
- The module exposes only `QTMiner`; `Attribute`, `Item`, `QTuple`, `Data`, `Cluster` and `ClusterSet` remain internal.

## qt_miner.py

This Python module is the entry point to the C++ core and is not a C++ class.

- On first import it computes a digest from all core `.cpp` sources and from the Python, pybind11 and compiler configuration.
- It compiles `bindings.cpp` with the available compiler in two steps (object file, then shared library) into a cache directory under the system temporary folder (for example `%TEMP%/qt_clustering/<digest>/` on Windows) and reuses the cached extension until the sources change.
- It loads `_qt_miner`, re-exports `QTMiner` and sets `QTMiner.__module__` to `src.core.qt_miner`; `__all__` contains only `QTMiner`.
