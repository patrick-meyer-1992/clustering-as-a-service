from sklearn.cluster import MiniBatchKMeans
from .base_clustering import BaseClustering
import collections

class MiniBatchKMeansClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, batch_size=1000, **params):
        super().__init__(dataset_name, columns)
        self.name = "Mini Batch KMeans"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params["batch_size"] = batch_size
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running MiniBatchKMeans with params: {self.params}")
            model = MiniBatchKMeans(**self.params)
            model.fit(data)
            
            result = {
                "labels": model.labels_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "n_iter": model.n_iter_
            }
            
            return result
            
        except Exception as e:
            print(f"Error in MiniBatchKMeans clustering: {str(e)}")
            raise





