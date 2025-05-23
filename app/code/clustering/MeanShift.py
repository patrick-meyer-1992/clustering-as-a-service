from sklearn.cluster import MeanShift
from .base_clustering import BaseClustering

class MeanShiftClustering(BaseClustering):
    def __init__(self, dataset_name, columns, bandwidth=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "mean_shift"
        if bandwidth is not None:
            self.params["bandwidth"] = bandwidth

    def run(self, data):
        model = MeanShift(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["cluster_centers_"] = model.cluster_centers_.tolist()
        return result


