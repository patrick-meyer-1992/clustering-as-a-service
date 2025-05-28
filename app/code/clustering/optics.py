from sklearn.cluster import OPTICS
from .base_clustering import BaseClustering
import collections
import numpy as np

class OPTICSClustering(BaseClustering):
    def __init__(self, dataset_name, columns, min_samples=5, max_eps=np.inf, **params):
        super().__init__(dataset_name, columns)
        self.name = "OPTICS"  # Exakt wie im Frontend
        self.params["min_samples"] = min_samples
        self.params["max_eps"] = max_eps
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running OPTICS with params: {self.params}")
            model = OPTICS(**self.params)
            model.fit(data)
            
            result = {
                "labels": model.labels_.tolist(),
                "reachability": model.reachability_.tolist(),
                "ordering": model.ordering_.tolist(),
                "core_distances": model.core_distances_.tolist(),
                "n_clusters": len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in OPTICS clustering: {str(e)}")
            raise






