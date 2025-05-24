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
        result = {}
        result["labels"] = model.labels_.tolist()
        result["centers"] = model.cluster_centers_.tolist()
        result["inertia"] = model.inertia_
        result["n_iter"] = model.n_iter_
        # TODO: Add more metrics if needed
        return result

