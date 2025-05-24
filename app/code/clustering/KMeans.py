from sklearn.cluster import KMeans
from .base_clustering import BaseClustering

class KMeansClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=3, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "kmeans"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = KMeans(**self.params)
        model.fit(data)

        labels = model.labels_

        import collections
        cluster_sizes = collections.Counter(labels)

        result = {
            "labels": labels.tolist(),
            "centers": model.cluster_centers_.tolist(),
            "inertia": model.inertia_,
            "n_iter": model.n_iter_,
            "n_clusters_": len(set(labels)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return result


