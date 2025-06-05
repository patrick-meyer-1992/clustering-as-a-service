from sklearn.cluster import Birch
from .base_clustering import BaseClustering
import collections

class BIRCHWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=3, threshold=0.5, **params):
        super().__init__(dataset_name, columns)
        self.name = "BIRCH"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params["threshold"] = threshold
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running BIRCH with params: {self.params}")
            model = Birch(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            
            result = {
                "labels": labels.tolist(),
                "n_clusters": len(set(labels)),
                "subcluster_centers": model.subcluster_centers_.tolist() if hasattr(model, 'subcluster_centers_') else None,
                "cluster_sizes": dict(cluster_sizes)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in BIRCH clustering: {str(e)}")
            raise






