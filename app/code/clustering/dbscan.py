from sklearn.cluster import DBSCAN
from .clustering_algorithm import ClusteringAlgorithm

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, data, eps=0.5, min_samples=5, preprocess=True, **kwargs):
        super().__init__(data, preprocess)
        self.eps = eps
        self.min_samples = min_samples
        self.kwargs = kwargs

    def run(self):
        X = self._prepare_data()
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, **self.kwargs)
        self.labels = self.model.fit_predict(X)
        # model_path, labels_path = self._save_results()
        self._save_results()

        return {
            "job_id": self.job_id,
            "algorithm": "dbscan",
            "model_path": "model_path",
            "labels_path": "labels_path"
        }
