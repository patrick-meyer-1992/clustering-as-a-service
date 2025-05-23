from sklearn.cluster import Birch
from .clustering_algorithm import ClusteringAlgorithm

class BIRCHClustering(ClusteringAlgorithm):
    def __init__(self, dataset_name, columns, threshold=0.5, n_clusters=3, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "birch"
        self.params["threshold"] = threshold
        self.params["n_clusters"] = n_clusters

    def run(self, data):
        model = Birch(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["subcluster_centers_"] = model.subcluster_centers_.tolist()
        result["n_subclusters_"] = model.subcluster_centers_.shape[0]
        return result





