from sklearn.cluster import AffinityPropagation
from .base_clustering import BaseClustering

class AffinityPropagationClustering(BaseClustering):
    def __init__(self, dataset_name, columns, damping=0.5, preference=None, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "affinity_propagation"
        self.params["damping"] = damping
        if preference is not None:
            self.params["preference"] = preference

    def run(self, data):
        model = AffinityPropagation(**self.params)
        model.fit(data)
        result = {}
        result["labels"] = model.labels_.tolist()
        result["cluster_centers_indices_"] = model.cluster_centers_indices_.tolist()
        result["cluster_centers_"] = model.cluster_centers_.tolist()
        result["n_iter_"] = model.n_iter_  # Kaç iterasyonda bitti
        return result

