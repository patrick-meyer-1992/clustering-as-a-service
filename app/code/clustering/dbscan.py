from sklearn.cluster import DBSCAN
from .base_clustering import BaseClustering

class DBSCANWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, eps=0.5, min_samples=5, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "dbscan"
        self.params["eps"] = eps
        self.params["min_samples"] = min_samples

    def run(self, data):
        model = DBSCAN(**self.params)
        model.fit(data)

        labels = model.labels_
        result = {
            "labels": labels.tolist(),
            "core_sample_indices_": model.core_sample_indices_.tolist(),
            "components_": model.components_.tolist(),
            "n_clusters_": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise_": list(labels).count(-1)
        }

        return result

