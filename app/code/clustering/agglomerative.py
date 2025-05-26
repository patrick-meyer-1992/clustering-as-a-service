from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from .base_clustering import BaseClustering
import collections

class AgglomerativeClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=2, linkage="ward", affinity="euclidean", **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "agglomerative"
        self.params["n_clusters"] = n_clusters
        self.params["linkage"] = linkage
        self.params["affinity"] = affinity

    def run(self, data):
        model = SklearnAgglomerativeClustering(**self.params)
        labels = model.fit_predict(data)

        cluster_sizes = collections.Counter(labels)

        result = {
            "labels": labels.tolist(),
            "n_clusters_": len(set(labels)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result





