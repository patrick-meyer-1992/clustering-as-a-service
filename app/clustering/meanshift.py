from sklearn.cluster import MeanShift
from .base_clustering import BaseClustering
import collections

class MeanShiftWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, bandwidth=None, **params):
        super().__init__(dataset_name, columns)
        self.name = "Mean Shift"  # Exakt wie im Frontend
        self.params["bandwidth"] = bandwidth
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running MeanShift with params: {self.params}")
            model = MeanShift(**self.params)
            model.fit(data)
            
            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}
            
            result = {
                "labels": model.labels_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_clusters": len(model.cluster_centers_),
                "cluster_sizes": dict(cluster_sizes)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in MeanShift clustering: {str(e)}")
            raise



