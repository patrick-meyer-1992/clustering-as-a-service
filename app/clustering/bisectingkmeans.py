from sklearn.cluster import BisectingKMeans
from .base_clustering import BaseClustering
import collections

class BisectingKMeansWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.name = "Bisecting KMeans"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running BisectingKMeans with params: {self.params}")
            model = BisectingKMeans(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            
            result = {
                "labels": labels.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "n_iter": model.n_iter_.tolist(),
                "cluster_sizes": dict(cluster_sizes)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in BisectingKMeans clustering: {str(e)}")
            raise







