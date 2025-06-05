from sklearn.cluster import OPTICS
from .base_clustering import BaseClustering
import collections

class OPTICSWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, min_samples=5, xi=0.05, min_cluster_size=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "optics"
        self.params["min_samples"] = min_samples
        self.params["xi"] = xi
        if min_cluster_size is not None:
            self.params["min_cluster_size"] = min_cluster_size

    def run(self, data):
        model = OPTICS(**self.params)
        model.fit(data)

        labels = model.labels_
        cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        result = {
            "labels": labels.tolist(),
            "reachability_": model.reachability_.tolist(),
            "core_distances_": model.core_distances_.tolist(),
            "ordering_": model.ordering_.tolist(),
            "n_clusters_": n_clusters,
            "n_noise_": n_noise,
            "cluster_sizes": dict(cluster_sizes)
        }

        return result






