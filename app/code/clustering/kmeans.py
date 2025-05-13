from sklearn.cluster import KMeans
from .clustering_algorithm import ClusteringAlgorithm
from sklearn.datasets import load_iris

class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, data, n_clusters=3, preprocess=True, **kwargs):
        super().__init__(data, preprocess)
        self.n_clusters = n_clusters
        self.kwargs = kwargs

    def run(self):
        # X = load_iris().data
        X = self._prepare_data()
        self.model = KMeans(n_clusters=self.n_clusters, **self.kwargs)
        self.labels = self.model.fit_predict(X)
        # model_path, labels_path = self._save_results()
        self._save_results()

        return {
            "job_id": self.job_id,
            "algorithm": "kmeans",
            "model_path": "model_path",
            "labels_path": "labels_path"
        }
