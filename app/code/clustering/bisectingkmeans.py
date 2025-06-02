from sklearn.cluster import BisectingKMeans
from .base_clustering import BaseClustering
import collections

class BisectingKMeansWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "bisecting_kmeans"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = BisectingKMeans(**self.params)
        model.fit(data)

        cluster_sizes = collections.Counter(model.labels_)

        result = {
            "labels": model.labels_.tolist(),
            "cluster_centers_": model.cluster_centers_.tolist(),
            "inertia": model.inertia_,
            "n_iter_": model.n_iter_,
            "n_clusters_": len(set(model.labels_)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result







