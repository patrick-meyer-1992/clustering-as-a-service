from sklearn.cluster import SpectralClustering
from .base_clustering import BaseClustering
import collections

class SpectralClusteringClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.name = "Spectral"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running Spectral clustering with params: {self.params}")
            model = SpectralClustering(**self.params)
            model.fit(data)
            
            result = {
                "labels": model.labels_.tolist(),
                "n_clusters": len(set(model.labels_)),
                "affinity_matrix": model.affinity_matrix_.tolist() if hasattr(model, 'affinity_matrix_') else None
            }
            
            return result
            
        except Exception as e:
            print(f"Error in Spectral clustering: {str(e)}")
            raise




