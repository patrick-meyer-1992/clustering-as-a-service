from sklearn.cluster import MiniBatchKMeans
from .base_clustering import BaseClustering

class MiniBatchKMeansClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, batch_size=100, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "minibatch_kmeans"
        self.params["n_clusters"] = n_clusters
        self.params["batch_size"] = batch_size

    def run(self, data):
        model = MiniBatchKMeans(**self.params)
        model.fit(data)

        import collections
        cluster_sizes = collections.Counter(model.labels_)

        result = {
            "labels": model.labels_.tolist(),
            "inertia": model.inertia_,
            "cluster_centers_": model.cluster_centers_.tolist(),
            "n_iter_": model.n_iter_,
            "n_clusters_": len(set(model.labels_)),
            "cluster_sizes": dict(cluster_sizes)
        }

        return return





