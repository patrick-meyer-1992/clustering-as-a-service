from sklearn.cluster import SpectralClustering
from .clustering_algorithm import ClusteringAlgorithm

class SpectralClustering(ClusteringAlgorithm):
    def __init__(self, dataset_name, columns, n_clusters=8, affinity='rbf', **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "spectral"
        self.params["n_clusters"] = n_clusters
        self.params["affinity"] = affinity

    def run(self, data):
        model = SpectralClustering(**self.params)
        labels = model.fit_predict(data)
        result = {}
        result["labels"] = labels.tolist()
        return result



