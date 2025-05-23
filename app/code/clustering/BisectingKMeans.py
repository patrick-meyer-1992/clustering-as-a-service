from sklearn.cluster import BisectingKMeans
from .base_clustering import BaseClustering

class BisectingKMeansClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "bisecting_kmeans"
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = BisectingKMeans(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["cluster_centers_"] = model.cluster_centers_.tolist()
        result["inertia"] = model.inertia_
        result["n_iter_"] = model.n_iter_
        return result






