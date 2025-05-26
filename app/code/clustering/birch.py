from sklearn.cluster import Birch
from .base_clustering import BaseClustering
import collections

class BIRCHClustering(BaseClustering):
    def __init__(self, dataset_name, columns, threshold=0.5, n_clusters=3, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "birch"
        self.params["threshold"] = threshold
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = Birch(**self.params)
        model.fit(data)

        labels = model.labels_
        cluster_sizes = collections.Counter(labels)

        result = {
            "labels": labels.tolist(),
            "subcluster_centers_": model.subcluster_centers_.tolist(),
            "n_subclusters_": model.subcluster_centers_.shape[0],
            "n_clusters_": len(set(labels)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result






