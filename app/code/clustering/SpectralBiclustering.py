from sklearn.cluster import SpectralBiclustering
from .base_clustering import BaseClustering

class SpectralBiclustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=(3, 3), **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "spectral_biclustering"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = SpectralBiclustering(**self.params)
        model.fit(data)

        result = {}
        result["rows_"] = [mask.tolist() for mask in model.rows_]
        result["columns_"] = [mask.tolist() for mask in model.columns_]
        result["shape"] = model.shape
        return result








