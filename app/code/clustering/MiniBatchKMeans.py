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
        result = {}
        result["labels"] = model.labels_.tolist()
        result["inertia"] = model.inertia_
        result["cluster_centers_"] = model.cluster_centers_.tolist()
        result["n_iter_"] = model.n_iter_
        return result




