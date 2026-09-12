# Core Module Description

This document briefly describes the role of every class defined in the `qt_clustering/core` folder.

### attributes.py
`ContinuousAttribute` and `DiscreteAttribute` describe the columns of a dataset: the former represent numeric features, the latter categorical ones.

### items.py
`Item` represents the value of a single feature of an example. `ContinuousItem` and `DiscreteItem` are its numeric and categorical variants, and know how to measure how different two values are.

### tuple.py
`QTuple` holds a whole example as a collection of `Item`s, and can compute the distance between two examples.

### data.py
`Data` wraps a dataset: its examples and the feature definitions. It loads data from CSV or NumPy arrays and converts them into the types the algorithm uses.

### cluster.py
`Cluster` groups together a set of clustered examples, with a reference example (the centroid) that identifies the group.

### cluster_set.py
`ClusterSet` is the collection of all clusters produced by the clustering algorithm.

### qt_miner.py
`QTMiner` implements the Quality Threshold algorithm: given a radius, it progressively partitions the dataset into clusters until every example is assigned to one.
