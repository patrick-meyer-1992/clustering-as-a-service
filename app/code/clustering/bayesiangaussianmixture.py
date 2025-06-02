from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from .base_clustering import BaseClustering
import collections

class BayesianGaussianMixtureWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, n_components=10, **params):
        super().__init__(dataset_name, columns, **params)
        self.name = "bayesian_gaussian_mixture"
        self.params["n_components"] = n_components
        self.transform_type = self.params.pop("transform_type", None)

    def prepare_data(self, data):
        if self.transform_type == "quantile":
            transformer = QuantileTransformer(output_distribution="normal")
            data = transformer.fit_transform(data)
        elif self.transform_type == "power":
            transformer = PowerTransformer()
            data = transformer.fit_transform(data)
        return data

    def run(self, data):
        data = self.prepare_data(data)

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

