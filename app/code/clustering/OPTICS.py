from sklearn.cluster import OPTICS
from .clustering_algorithm import ClusteringAlgorithm

class OPTICSClustering(ClusteringAlgorithm):
    def __init__(self, dataset_name, columns, min_samples=5, xi=0.05, min_cluster_size=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "optics"
        self.params["min_samples"] = min_samples
        self.params["xi"] = xi
        if min_cluster_size is not None:
            self.params["min_cluster_size"] = min_cluster_size

    def run(self, data):
        model = OPTICS(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["reachability_"] = model.reachability_.tolist()
        result["core_distances_"] = model.core_distances_.tolist()
        result["ordering_"] = model.ordering_.tolist()
        return result





