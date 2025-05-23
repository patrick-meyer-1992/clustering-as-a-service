from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from .clustering_algorithm import ClusteringAlgorithm

class AgglomerativeClustering(ClusteringAlgorithm):
    def __init__(self, dataset_name, columns, n_clusters=2, linkage="ward", affinity="euclidean", **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "agglomerative"
        self.params["n_clusters"] = n_clusters
        self.params["linkage"] = linkage
        self.params["affinity"] = affinity

    def run(self, data):
        model = SklearnAgglomerativeClustering(**self.params)
        labels = model.fit_predict(data)
        result = {}
        result["labels"] = labels.tolist()
        return result




