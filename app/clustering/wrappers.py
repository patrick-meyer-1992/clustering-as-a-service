import collections

import numpy as np
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .base_clustering import BaseClustering


class AffinityPropagationWrapper(BaseClustering):
    backend_name = "affinitypropagation"
    frontend_name = "Affinity Propagation"

    def __init__(self, dataset_name, columns, **params):
        super().__init__(dataset_name, columns)
        self.params = {
            "damping": params.get("damping", 0.5),
            "max_iter": params.get("max_iter", 200),
            "convergence_iter": params.get("convergence_iter", 15),
            "copy": params.get("copy", True),
            "preference": params.get("preference"),
            "affinity": params.get("affinity", "euclidean"),
            "verbose": params.get("verbose", False),
        }

    @staticmethod
    def get_default_params():
        return AffinityPropagation().get_params()

    def run(self, data):
        try:
            print(f"Running AffinityPropagation with params: {self.params}")
            model = AffinityPropagation(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "cluster_centers_indices": model.cluster_centers_indices_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_iterations": model.n_iter_,
                "n_clusters": len(model.cluster_centers_indices_),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class AgglomerativeClusteringWrapper(BaseClustering):
    backend_name = "agglomerative"
    frontend_name = "Agglomerative"

    def __init__(
        self, dataset_name, columns, n_clusters=2, linkage="ward", affinity="euclidean", **params
    ):
        super().__init__(dataset_name, columns, **params)
        self.params["n_clusters"] = n_clusters
        self.params["linkage"] = linkage
        self.params["affinity"] = affinity

    @staticmethod
    def get_default_params():
        return AgglomerativeClustering().get_params()

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
                "n_leaves": model.n_leaves_ if hasattr(model, "n_leaves_") else None,
                "distances": model.distances_.tolist() if hasattr(model, "distances_") else None,
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BayesianGaussianMixtureWrapper(BaseClustering):
    backend_name = "bayesiangaussianmixture"
    frontend_name = "Bayesian Gaussian Mixture"

    def __init__(self, dataset_name, columns, n_components=10, **params):
        super().__init__(dataset_name, columns, **params)
        self.params["n_components"] = n_components
        self.transform_type = self.params.pop("transform_type", None)

    @staticmethod
    def get_default_params():
        return BayesianGaussianMixture().get_params()

    def run(self, data):
        try:
            model = BayesianGaussianMixture(**self.params)
            model.fit(data)

            labels = model.predict(data)
            probabilities = model.predict_proba(data)

            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "probabilities": probabilities.tolist(),
                "means": model.means_.tolist(),
                "covariances": model.covariances_.tolist(),
                "n_iter": model.n_iter_,
                "n_clusters_": len(set(labels)),
                "cluster_sizes": dict(cluster_sizes),
                "weights": model.weights_.tolist(),
            }

            return result
        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BIRCHWrapper(BaseClustering):
    backend_name = "birch"
    frontend_name = "BIRCH"

    def __init__(self, dataset_name, columns, n_clusters=3, threshold=0.5, **params):
        super().__init__(dataset_name, columns)
        self.params["n_clusters"] = n_clusters
        self.params["threshold"] = threshold
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return Birch().get_params()

    def run(self, data):
        try:
            print(f"Running BIRCH with params: {self.params}")
            model = Birch(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "n_clusters": len(set(labels)),
                "subcluster_centers": model.subcluster_centers_.tolist()
                if hasattr(model, "subcluster_centers_")
                else None,
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BisectingKMeansWrapper(BaseClustering):
    backend_name = "bisectingkmeans"
    frontend_name = "Bisecting KMeans"

    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return BisectingKMeans().get_params()

    def run(self, data):
        try:
            print(f"Running BisectingKMeans with params: {self.params}")
            model = BisectingKMeans(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "n_iter": model.n_iter_.tolist(),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class DBSCANWrapper(BaseClustering):
    backend_name = "dbscan"
    frontend_name = "DBSCAN"

    def __init__(self, dataset_name, columns, eps=0.5, min_samples=5, **params):
        super().__init__(dataset_name, columns)
        self.params["eps"] = eps
        self.params["min_samples"] = min_samples
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return DBSCAN().get_params()

    def run(self, data):
        try:
            print(f"Running DBSCAN with params: {self.params}")
            model = DBSCAN(**self.params)
            model.fit(data)

            result = {
                "labels": model.labels_.tolist(),
                "n_clusters": len(set(model.labels_)) - (1 if -1 in model.labels_ else 0),
                "n_noise": list(model.labels_).count(-1),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class GaussianMixtureWrapper(BaseClustering):
    backend_name = "gaussianmixture"
    frontend_name = "Gaussian Mixture"

    def __init__(self, dataset_name, columns, n_components=3, **params):
        super().__init__(dataset_name, columns, **params)
        self.params["n_components"] = n_components
        self.transform_type = self.params.pop("transform_type", None)

    @staticmethod
    def get_default_params():
        return GaussianMixture().get_params()

    def run(self, data):
        try:
            model = GaussianMixture(**self.params)
            model.fit(data)

            labels = model.predict(data)
            probabilities = model.predict_proba(data)

            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "probabilities": probabilities.tolist(),
                "means": model.means_.tolist(),
                "covariances": model.covariances_.tolist(),
                "n_iter": model.n_iter_,
                "n_clusters_": len(set(labels)),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result
        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class KMeansWrapper(BaseClustering):
    backend_name = "kmeans"
    frontend_name = "KMeans"

    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.params["n_clusters"] = n_clusters
        self.params.update(params)  # Weitere Parameter hinzufügen

    @staticmethod
    def get_default_params():
        return KMeans().get_params()

    def run(self, data):
        try:
            print(f"Running KMeans with params: {self.params}")
            model = KMeans(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": model.labels_.tolist(),
                "centers": model.cluster_centers_.tolist(),
                "n_iter": model.n_iter_,
                "inertia": float(model.inertia_),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class MeanShiftWrapper(BaseClustering):
    backend_name = "meanshift"
    frontend_name = "Mean Shift"

    def __init__(self, dataset_name, columns, bandwidth=None, **params):
        super().__init__(dataset_name, columns)
        self.params["bandwidth"] = bandwidth
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return MeanShift().get_params()

    def run(self, data):
        try:
            print(f"Running MeanShift with params: {self.params}")
            model = MeanShift(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}

            result = {
                "labels": labels.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_clusters": len(model.cluster_centers_),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class MiniBatchKMeansWrapper(BaseClustering):
    backend_name = "minibatchkmeans"
    frontend_name = "Mini Batch KMeans"

    def __init__(self, dataset_name, columns, n_clusters=8, batch_size=1000, **params):
        super().__init__(dataset_name, columns)
        self.params["n_clusters"] = n_clusters
        self.params["batch_size"] = batch_size
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return MiniBatchKMeans().get_params()

    def run(self, data):
        try:
            print(f"Running MiniBatchKMeans with params: {self.params}")
            model = MiniBatchKMeans(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": model.labels_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "n_iter": model.n_iter_,
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class OPTICSWrapper(BaseClustering):
    backend_name = "optics"
    frontend_name = "OPTICS"

    def __init__(self, dataset_name, columns, min_samples=5, max_eps=np.inf, **params):
        super().__init__(dataset_name, columns)
        self.params["min_samples"] = min_samples
        self.params["max_eps"] = max_eps
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return OPTICS().get_params()

    def run(self, data):
        try:
            print(f"Running OPTICS with params: {self.params}")
            model = OPTICS(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "reachability": model.reachability_.tolist(),
                "ordering": model.ordering_.tolist(),
                "core_distances": model.core_distances_.tolist(),
                "n_clusters": len(set(labels)) - (1 if -1 in model.labels_ else 0),
                "n_noise_": list(labels).count(-1),
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class SpectralClusteringWrapper(BaseClustering):
    backend_name = "spectralclustering"
    frontend_name = "Spectral Clustering"

    def __init__(self, dataset_name, columns, n_clusters=8, **params):
        super().__init__(dataset_name, columns)
        self.params["n_clusters"] = n_clusters
        self.params.update(params)

    @staticmethod
    def get_default_params():
        return SpectralClustering().get_params()

    def run(self, data):
        try:
            print(f"Running Spectral clustering with params: {self.params}")
            model = SpectralClustering(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}

            result = {
                "labels": labels.tolist(),
                "n_clusters": len(set(labels)),
                "affinity_matrix": model.affinity_matrix_.tolist()
                if hasattr(model, "affinity_matrix_")
                else None,
                "cluster_sizes": dict(cluster_sizes),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e
