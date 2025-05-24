from sklearn.cluster import DBSCAN
from .base_clustering import BaseClustering

class DBSCANClustering(BaseClustering):
    def __init__(self, dataset_name, columns, eps=0.5, min_samples=5, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "dbscan"
        self.params["eps"] = eps
        self.params["min_samples"] = min_samples

    def run(self, data):
        model = DBSCAN(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["core_sample_indices_"] = model.core_sample_indices_.tolist()
        result["components_"] = model.components_.tolist() 
        # TODO: Add more metrics if needed
        return result
