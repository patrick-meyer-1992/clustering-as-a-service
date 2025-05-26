from sklearn.cluster import SpectralBiclustering
from .base_clustering import BaseClustering

class SpectralBiclusteringClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=(3, 3), **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "spectral_biclustering"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = SpectralBiclustering(**self.params)
        model.fit(data)

        rows = model.rows_
        cols = model.columns_

        bicluster_sizes = [
            int(r.sum()) * int(c.sum()) for r, c in zip(rows, cols)
        ]

        result = {
            "rows_": [r.tolist() for r in rows],
            "columns_": [c.tolist() for c in cols],
            "shape": model.shape,
            "n_biclusters_": len(rows),
            "bicluster_sizes": bicluster_sizes
        }

        return result









