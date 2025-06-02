from sklearn.mixture import BayesianGaussianMixture
from .base_clustering import BaseClustering
import collections


class BayesianGaussianMixtureClustering(BaseClustering):
    def __init__(self, dataset_name, columns, n_components=10, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "bayesian_gaussian_mixture"
        self.params["n_components"] = n_components

    def run(self, data):
        model = BayesianGaussianMixture(**self.params)
        model.fit(data)

        labels = model.predict(data)
        probabilities = model.predict_proba(data)

        cluster_sizes = collections.Counter(labels)

        result = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "means": model.means_.tolist(),
            "covariances": model.covariances_.tolist(),
            "n_iter": model.n_iter_,
            "n_clusters_": len(set(labels)),
            "cluster_sizes": dict(cluster_sizes),
            "weights": model.weights_.tolist()
        }

        return result
