from sklearn.cluster import SpectralCoclustering
from .base_clustering import BaseClustering
import collections

class SpectralCoclusteringClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=3, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "spectral_coclustering"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = SpectralCoclustering(**self.params)
        model.fit(data)

        rows = model.rows_
        cols = model.columns_

        cocluster_sizes = [
            int(r.sum()) * int(c.sum()) for r, c in zip(rows, cols)
        ]

        result = {
            "rows_": [r.tolist() for r in rows],
            "columns_": [c.tolist() for c in cols],
            "shape": model.shape,
            "n_coclusters_": len(rows),
            "cocluster_sizes": cocluster_sizes
        }

        return result










