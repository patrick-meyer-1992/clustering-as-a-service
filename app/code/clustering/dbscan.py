from sklearn.cluster import DBSCAN
from .base_clustering import BaseClustering

class DBSCANClustering(BaseClustering):
    def __init__(self, dataset_name, columns, eps=0.5, min_samples=5, **params):
        super().__init__(dataset_name, columns)
        self.name = "DBSCAN"  # Exakt wie im Frontend
        self.params["eps"] = eps
        self.params["min_samples"] = min_samples
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running DBSCAN with params: {self.params}")
            model = DBSCAN(**self.params)
            model.fit(data)
            
            result = {
                "labels": model.labels_.tolist(),
                "n_clusters": len(set(model.labels_)) - (1 if -1 in model.labels_ else 0),
                "n_noise": list(model.labels_).count(-1)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in DBSCAN clustering: {str(e)}")
            raise

