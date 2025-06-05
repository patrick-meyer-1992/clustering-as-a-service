from sklearn.cluster import MeanShift
from .base_clustering import BaseClustering
import collections

class MeanShiftWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, bandwidth=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "mean_shift"
        if bandwidth is not None:
            self.params["bandwidth"] = bandwidth

    def run(self, data):
        model = MeanShift(**self.params)
        model.fit(data)

        labels = model.labels_
        cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}

        result = {
            "labels": labels.tolist(),
            "cluster_centers_": model.cluster_centers_.tolist(),
            "n_clusters_": len(model.cluster_centers_),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result



