from sklearn.cluster import AgglomerativeClustering
from .base_clustering import BaseClustering
import collections

class AgglomerativeClusteringWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=2, linkage='ward', **params):
        super().__init__(dataset_name, columns)
        self.name = "Agglomerative"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params["linkage"] = linkage
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running Agglomerative with params: {self.params}")
            model = AgglomerativeClustering(**self.params)
            model.fit(data)
            labels = model.fit_predict(data)
            
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": model.labels_.tolist(),
                "n_clusters": model.n_clusters_,
                "n_leaves": model.n_leaves_ if hasattr(model, 'n_leaves_') else None,
                "distances": model.distances_.tolist() if hasattr(model, 'distances_') else None,
                "cluster_sizes": dict(cluster_sizes)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in Agglomerative clustering: {str(e)}")
            raise





