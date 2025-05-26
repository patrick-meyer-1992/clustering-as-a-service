from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from .base_clustering import BaseClustering
import collections

class FeatureAgglomerationClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_clusters=5, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "feature_agglomeration"
        self.params["n_clusters"] = n_clusters

    def prepare_data(self, data, preprocess):
        X = data
        if preprocess:
            scaler = StandardScaler()
            X = scaler.fit_transform(X.T).T  # Normalize columns (features), not rows (samples)
        return X

    def run(self, data):
        model = FeatureAgglomeration(**self.params)
        model.fit(data)
        transformed = model.transform(data)

        result = {
            "transformed": transformed.tolist(),
            "n_features_in_": model.n_features_in_,
            "n_output_features_": transformed.shape[1]
        }

        return result








