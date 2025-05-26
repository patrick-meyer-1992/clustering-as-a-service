from sklearn.cluster import AffinityPropagation
from .base_clustering import BaseClustering
import collections


class AffinityPropagationClustering(BaseClustering):
    def __init__(self, dataset_name, columns, damping=0.5, preference=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "affinity_propagation"
        self.params["damping"] = damping
        if preference is not None:
            self.params["preference"] = preference

    def run(self, data):
        model = AffinityPropagation(**self.params)
        model.fit(data)

        cluster_sizes = collections.Counter(model.labels_)

        result = {
            "labels": model.labels_.tolist(),
            "cluster_centers_indices_": model.cluster_centers_indices_.tolist(),
            "cluster_centers_": model.cluster_centers_.tolist(),
            "n_iter_": model.n_iter_,
            "n_clusters_": len(model.cluster_centers_indices_),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result


