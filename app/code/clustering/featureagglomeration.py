from sklearn.cluster import FeatureAgglomeration
from .base_clustering import BaseClustering
import collections

class FeatureAgglomerationClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=2, **params):
        super().__init__(dataset_name, columns)
        self.name = "Feature Agglomeration"  # Exakt wie im Frontend
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    def run(self, data):
        try:
            print(f"Running FeatureAgglomeration with params: {self.params}")
            model = FeatureAgglomeration(**self.params)
            model.fit(data)
            
            result = {
                "labels": model.labels_.tolist(),
                "n_clusters": model.n_clusters_,
                "n_leaves": model.n_leaves_,
                "children": model.children_.tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"Error in FeatureAgglomeration clustering: {str(e)}")
            raise








