from sklearn.cluster import SpectralClustering
from .base_clustering import BaseClustering
import collections

class SpectralClusteringWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, affinity='rbf', **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "spectral"
        self.params["n_clusters"] = n_clusters
        self.params["affinity"] = affinity

    def run(self, data):
        model = SpectralClustering(**self.params)
        labels = model.fit_predict(data)

        cluster_sizes = collections.Counter(labels)

        result = {
            "labels": labels.tolist(),
            "n_clusters_": len(set(labels)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result




