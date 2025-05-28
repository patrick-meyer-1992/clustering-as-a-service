from sklearn.cluster import SpectralBiclustering
from .base_clustering import BaseClustering
import collections

class SpectralBiclusteringClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=3, **params):
        super().__init__(dataset_name, columns)
        self.name = "Spectral Biclustering"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running SpectralBiclustering with params: {self.params}")
            model = SpectralBiclustering(**self.params)
            model.fit(data)
            
            result = {
                "row_labels": model.row_labels_.tolist(),
                "column_labels": model.column_labels_.tolist(),
                "rows": model.rows_.tolist(),
                "columns": model.columns_.tolist(),
                "n_clusters": model.n_clusters
            }
            
            return result
            
        except Exception as e:
            print(f"Error in SpectralBiclustering clustering: {str(e)}")
            raise









