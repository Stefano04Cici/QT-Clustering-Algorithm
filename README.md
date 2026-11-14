# How Quality Threshold (QT) Clustering Works

This is a deterministic, partitioning clustering algorithm designed to group data points based on a user-defined quality constraint (maximum cluster diameter).
Unlike algorithms like K-Means, QT Clustering does not require you to specify the number of clusters in advance.

---

## Key Concept
The algorithm guarantees that **no cluster will exceed a specified maximum distance (threshold)** among its points. This ensures high-quality, highly coherent clusters.

---

## Limitations

The algorithm is highly inefficient for large datasets because it must evaluate the cluster diameter constraint for **every single data point** to build candidate clusters. This leads to a quadratic time complexity $O(n^2)$ and redundant distance calculations.

---

## Step-by-Step Mechanism

1. **Set the Threshold ($\mathbf{d_{\text{max}}}$):**
   Define the maximum allowable diameter (or distance) for any cluster.

2. **Build Candidate Clusters:**
   For **every** point in the dataset, create a candidate cluster:
   - Start with the point itself.
   - Iteratively add the closest available point that keeps the cluster's diameter $\le d_{\text{max}}$.
   - Continue adding points until no more points can be included without exceeding the threshold.

3. **Select the Best Cluster:**
   Find the candidate cluster with the largest number of points (i.e., maximum cardinality). This becomes an official cluster.

4. **Update and Repeat:**
   - Remove all points of the selected cluster from the dataset.
   - Repeat steps 2 and 3 for all remaining points.

5. **Termination:**
    The algorithm stops when all points are assigned to a cluster or isolated as noise.

---

## Setup and Run

### Prerequisites
- Python 3.13 or higher
- A C++17 compiler:
  - Windows: MinGW-w64 `g++` (e.g. MSYS2 UCRT64); MSVC `cl.exe` is not supported
  - Linux: `g++` / `build-essential`
  - macOS: Xcode Command Line Tools
- Python development headers and import library (included in uv-managed CPython; on Linux with a system Python, install `python3-dev`)
- `uv` (optional, only for the uv workflow)

### Venv

Venv creation with Python:

```bash
python -m venv .venv
```
```bash
source .venv/bin/activate    # Linux
```
```bash
.venv\Scripts\activate       # Windows
```

Venv creation with uv:

```bash
uv venv .venv
```
```bash
source .venv/bin/activate    # Linux/macOS
```
```bash
.venv\Scripts\activate       # Windows
```

### Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Install the required dependencies using uv:

```bash
uv sync
```

### Running
Start the program in interactive mode:

```bash
python main.py
```

```bash
uv run main.py
```
