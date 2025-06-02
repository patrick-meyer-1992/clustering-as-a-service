# 🧠 Clustering Algorithms - Wrapper Framework

A modular Python framework that wraps all major clustering algorithms from `scikit-learn`, providing a unified interface for loading, preprocessing, running, and saving results via API.

---

## 📂 Project Structure

clustering/
├── base_clustering.py                # Unified abstract base class for row-based clustering
├── affinitypropagation.py
├── agglomerative.py
├── bayesiangaussianmixture.py
├── birch.py
├── bisectingkmeans.py
├── dbscan.py
├── gaussianmixture.py
├── kmeans.py
├── meanshift.py
├── minibatchkmeans.py
├── optics.py
├── spectral.py
├── column_based/                     # For non-row-based clustering methods
│   ├── featureagglomeration.py       # Clusters features (columns), not instances
│   ├── spectralbiclustering.py       # Clusters rows and columns simultaneously
│   └── spectralcoclustering.py       # Similar to biclustering with matrix factorization

---

## 🔧 Base Class: `BaseClustering`

Located in `base_clustering.py`, this abstract class provides common functionality for row-based clustering algorithms:

- `load_data()` – Fetch dataset via FastAPI
- `prepare_data()` – Scalable preprocessing pipeline supporting:
  - Multiple scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`)
  - Optional normalization (`L1`, `L2`)
  - Optional PCA dimensionality reduction
- `run()` – Abstract method each subclass must implement
- `save_results()` – Send results to FastAPI with timestamps and metadata

---

## 📌 Available Clustering Wrappers

### ✅ Row-Based Clustering (compatible with `BaseClustering`)

| Algorithm                   | Class Name                              | Notes                                  |
|----------------------------|-----------------------------------------|----------------------------------------|
| KMeans                     | `KMeansClustering`                      | Classic partitioning                   |
| MiniBatchKMeans            | `MiniBatchKMeansClustering`             | Faster KMeans variant                  |
| BisectingKMeans            | `BisectingKMeansClustering`             | Hierarchical + partitioning            |
| DBSCAN                     | `DBSCANClustering`                      | Density-based, no `n_clusters` needed  |
| OPTICS                     | `OPTICSClustering`                      | Similar to DBSCAN but hierarchical     |
| MeanShift                  | `MeanShiftClustering`                   | Bandwidth-sensitive                    |
| AffinityPropagation        | `AffinityPropagationClustering`         | Does not require `n_clusters`          |
| AgglomerativeClustering    | `AgglomerativeClustering`               | Linkage-based hierarchical method      |
| Birch                      | `BIRCHClustering`                       | Balanced hierarchical clustering       |
| SpectralClustering         | `SpectralClusteringClustering`          | Graph-based clustering                 |
| GaussianMixture            | `GaussianMixtureClustering`             | Soft probabilistic clustering          |
| BayesianGaussianMixture    | `BayesianGaussianMixtureClustering`     | Infinite mixture model with variational Bayes |

### ⚠️ Column/Matrix-Based Clustering (incompatible with `BaseClustering`)

| Algorithm                 | File Location                         | Notes                                           |
|--------------------------|---------------------------------------|-------------------------------------------------|
| FeatureAgglomeration     | `column_based/featureagglomeration.py` | Clusters features (columns), not instances      |
| SpectralBiclustering     | `column_based/spectralbiclustering.py` | Simultaneous row and column clustering          |
| SpectralCoclustering     | `column_based/spectralcoclustering.py` | Similar to biclustering with matrix factorization|

---

## 🧪 Usage

Each class expects:
- `dataset_name`: string identifier
- `columns`: list of feature names
- `**params`: any algorithm-specific parameters

Example:

```python
from kmeans import KMeansClustering

model = KMeansClustering(dataset_name="iris", columns=["sepal_length", "sepal_width"], n_clusters=3)
X = model.prepare_data(model.load_data(), preprocess=True)
result = model.run(X)
model.save_results(result, job_id="abc123", created_timestamp=..., started_timestamp=..., user_id=42)
```
