from sklearn.cluster import KMeans
from .base_clustering import BaseClustering
import collections


class KMeansClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.name = "KMeans"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params.update(params)  # Weitere Parameter hinzufügen

    def run(self, data):
        try:
            print(f"Running KMeans with params: {self.params}")
            model = KMeans(**self.params)
            model.fit(data)

            result = {
                "labels": model.labels_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_iter": model.n_iter_,
                "inertia": float(model.inertia_)
            }

            return result

        except Exception as e:
            print(f"Error in KMeans clustering: {str(e)}")
            raise


