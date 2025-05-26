# 🧠 Clustering Algorithms - Wrapper Framework

A modular Python framework that wraps all major clustering algorithms from `scikit-learn`, providing a unified interface for loading, preprocessing, running, and saving results via API.

---

## 📂 Project Structure

clustering/
├── base_clustering.py # Abstract base class for all clustering models
├── AffinityPropagation.py
├── AgglomerativeClustering.py
├── BIRCH.py
├── BisectingKMeans.py
├── DBSCAN.py
├── FeatureAgglomeration.py
├── KMeans.py
├── MeanShift.py
├── MiniBatchKMeans.py
├── OPTICS.py
├── SpectralClustering.py
├── SpectralBiclustering.py
└── SpectralCoclustering.py

---

## 🔧 Base Class: `BaseClustering`

Located in `base_clustering.py`, this abstract class provides common functionality:

- `load_data()` – Fetch dataset via FastAPI
- `prepare_data()` – Standard preprocessing with `StandardScaler`
- `run()` – Abstract method each subclass must implement
- `save_results()` – Send results to FastAPI with timestamps and metadata

---

## 📌 Available Clustering Wrappers

| Algorithm                 | Class Name                        | Notes                                  |
|---------------------------|------------------------------------|----------------------------------------|
| KMeans                    | `KMeansClustering`                | Classic partitioning                   |
| MiniBatchKMeans           | `MiniBatchKMeansClustering`       | Faster KMeans variant                  |
| BisectingKMeans           | `BisectingKMeansClustering`       | Hierarchical + partitioning            |
| DBSCAN                    | `DBSCANClustering`                | Density-based, no `n_clusters` needed  |
| OPTICS                    | `OPTICSClustering`                | Similar to DBSCAN but hierarchical     |
| MeanShift                 | `MeanShiftClustering`             | Bandwidth-sensitive                    |
| AffinityPropagation       | `AffinityPropagationClustering`   | Does not require `n_clusters`          |
| AgglomerativeClustering   | `AgglomerativeClustering`         | Linkage-based hierarchical method      |
| Birch                     | `BIRCHClustering`                 | Balanced hierarchical clustering       |
| SpectralClustering        | `SpectralClustering`              | Graph-based clustering                 |
| FeatureAgglomeration      | `FeatureAgglomerationClustering`  | Feature-based dimensionality reduction |
| SpectralBiclustering      | `SpectralBiclusteringClustering`  | Simultaneous row and column clustering |
| SpectralCoclustering      | `SpectralCoclusteringClustering`  | Similar to biclustering, different math|

---

## 🧪 Usage

Each class expects:
- `dataset_name`: string identifier
- `columns`: list of feature names
- `**params`: any algorithm-specific parameters

Example:

```python
from KMeans import KMeansClustering

model = KMeansClustering(dataset_name="iris", columns=["sepal_length", "sepal_width"], n_clusters=3)
X = model.prepare_data(model.load_data(), preprocess=True)
result = model.run(X)
model.save_results(result, job_id="abc123", created_timestamp=..., started_timestamp=..., user_id=42)
