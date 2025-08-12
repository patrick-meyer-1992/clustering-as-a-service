import collections

from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
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
    """
    Wrapper for the Affinity Propagation algorithm.
    Identifies exemplars among data points and forms clusters.
    """

    backend_name = "affinitypropagation"
    frontend_name = "Affinity Propagation"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return AffinityPropagation().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return AffinityPropagation

    def run(self, data):
        """
        Run Affinity Propagation and return labels, cluster centers, and quality metrics.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = AffinityPropagation(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "cluster_centers_indices": model.cluster_centers_indices_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_iterations": model.n_iter_,
                "n_clusters": len(model.cluster_centers_indices_),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class AgglomerativeClusteringWrapper(BaseClustering):
    """
    Wrapper for Agglomerative (hierarchical) clustering.
    Forms nested clusters by merging or splitting.
    """

    backend_name = "agglomerative"
    frontend_name = "Agglomerative"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return AgglomerativeClustering().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return AgglomerativeClustering

    def run(self, data):
        """
        Apply agglomerative clustering and return labels and linkage info.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = AgglomerativeClustering(**self.clustering_params)
            model.fit(data)

            labels = model.fit_predict(data)
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "n_clusters": model.n_clusters_,
                "n_leaves": model.n_leaves_ if hasattr(model, "n_leaves_") else None,
                "distances": model.distances_.tolist() if hasattr(model, "distances_") else None,
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BayesianGaussianMixtureWrapper(BaseClustering):
    """
    Wrapper for Bayesian Gaussian Mixture Model.
    Allows automatic complexity control via priors.
    """

    backend_name = "bayesiangaussianmixture"
    frontend_name = "Bayesian Gaussian Mixture"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return BayesianGaussianMixture().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return BayesianGaussianMixture

    def run(self, data):
        """
        Fit BGM model and return probabilistic labels and component stats.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")

            reg_covar = self.clustering_params.get("reg_covar")
            if isinstance(reg_covar, str):
                try:
                    self.clustering_params["reg_covar"] = float(reg_covar)
                except ValueError:
                    raise RuntimeError(f"Invalid reg_covar value: {reg_covar}")

            model = BayesianGaussianMixture(**self.clustering_params)
            model.fit(data)

            labels = model.predict(data)
            probabilities = model.predict_proba(data)
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "probabilities": probabilities.tolist(),
                "means": model.means_.tolist(),
                "covariances": model.covariances_.tolist(),
                "n_iter": model.n_iter_,
                "n_clusters_": len(set(labels)),
                "cluster_sizes": dict(cluster_sizes),
                "weights": model.weights_.tolist(),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result
        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BIRCHWrapper(BaseClustering):
    """
    Wrapper for the BIRCH algorithm.
    Scales well for large datasets by building CF trees.
    """

    backend_name = "birch"
    frontend_name = "BIRCH"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return Birch().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return Birch

    def run(self, data):
        """
        Run BIRCH clustering and return subcluster centers and metrics.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = Birch(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "n_clusters": len(set(labels)),
                "subcluster_centers": model.subcluster_centers_.tolist()
                if hasattr(model, "subcluster_centers_")
                else None,
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class BisectingKMeansWrapper(BaseClustering):
    """
    Wrapper for Bisecting KMeans.
    Improves KMeans by recursive splitting.
    """

    backend_name = "bisectingkmeans"
    frontend_name = "Bisecting KMeans"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return BisectingKMeans().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return BisectingKMeans

    def run(self, data):
        """
        Apply Bisecting KMeans and return centers and inertia.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = BisectingKMeans(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class DBSCANWrapper(BaseClustering):
    """
    Wrapper for DBSCAN.
    Density-based clustering with noise handling.
    """

    backend_name = "dbscan"
    frontend_name = "DBSCAN"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return DBSCAN().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return DBSCAN

    def run(self, data):
        """
        Run DBSCAN and return core samples, labels, and cluster info.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = DBSCAN(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "n_clusters": len(set(model.labels_)) - (1 if -1 in model.labels_ else 0),
                "n_noise": list(model.labels_).count(-1),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class GaussianMixtureWrapper(BaseClustering):
    """
    Wrapper for Gaussian Mixture Model.
    Uses soft probabilistic clustering.
    """

    backend_name = "gaussianmixture"
    frontend_name = "Gaussian Mixture"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return GaussianMixture().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return GaussianMixture

    def run(self, data):
        """
        Fit GMM and return probabilities, means, and covariances.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = GaussianMixture(**self.clustering_params)
            model.fit(data)

            labels = model.predict(data)
            probabilities = model.predict_proba(data)
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "probabilities": probabilities.tolist(),
                "means": model.means_.tolist(),
                "covariances": model.covariances_.tolist(),
                "n_iter": model.n_iter_,
                "n_clusters_": len(set(labels)),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result
        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class HDBSCANWrapper(BaseClustering):
    """
    Wrapper for HDBSCAN.
    Hierarchical density-based clustering with soft cluster probability.
    """

    backend_name = "hdbscan"
    frontend_name = "HDBSCAN"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return HDBSCAN().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return HDBSCAN

    def run(self, data):
        """
        Apply HDBSCAN and return labels, soft probabilities, and centroids if available.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = HDBSCAN(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "n_clusters": len([label for label in set(labels) if label >= 0]),  # exclude -1, -2, -3
                "n_noise": list(labels).count(-1),
                "cluster_sizes": dict(cluster_sizes),
                "probabilities": model.probabilities_.tolist() if hasattr(model, "probabilities_") else None,
                "centroids": model.centroids_.tolist() if hasattr(model, "centroids_") else None,
                "medoids": model.medoids_.tolist() if hasattr(model, "medoids_") else None,
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class KMeansWrapper(BaseClustering):
    """
    Wrapper for KMeans algorithm.
    Standard centroid-based clustering.
    """

    backend_name = "kmeans"
    frontend_name = "KMeans"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return KMeans().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return KMeans

    def run(self, data):
        """
        Run KMeans and return cluster centers and inertia.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = KMeans(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "centers": model.cluster_centers_.tolist(),
                "n_iter": model.n_iter_,
                "inertia": float(model.inertia_),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class MeanShiftWrapper(BaseClustering):
    """
    Wrapper for Mean Shift.
    Uses kernel density estimates to discover clusters.
    """

    backend_name = "meanshift"
    frontend_name = "Mean Shift"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return MeanShift().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return MeanShift

    def run(self, data):
        """
        Run Mean Shift and return cluster centers and number of clusters.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = MeanShift(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_clusters": len(model.cluster_centers_),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class MiniBatchKMeansWrapper(BaseClustering):
    """
    Wrapper for Mini Batch KMeans.
    Faster variant of KMeans using small random batches.
    """

    backend_name = "minibatchkmeans"
    frontend_name = "Mini Batch KMeans"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return MiniBatchKMeans().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return MiniBatchKMeans

    def run(self, data):
        """
        Run MiniBatch KMeans and return centers, inertia, and iterations.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = MiniBatchKMeans(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": float(model.inertia_),
                "n_iter": model.n_iter_,
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class OPTICSWrapper(BaseClustering):
    """
    Wrapper for OPTICS algorithm.
    Orders points for cluster structure and extracts density-based clusters.
    """

    backend_name = "optics"
    frontend_name = "OPTICS"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        params = OPTICS().get_params()
        return params

    @staticmethod
    def get_sklearn_estimator_class():
        return OPTICS

    def run(self, data):
        """
        Run OPTICS and return reachability and cluster assignments.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = OPTICS(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "reachability": model.reachability_.tolist(),
                "ordering": model.ordering_.tolist(),
                "core_distances": model.core_distances_.tolist(),
                "n_clusters": len(set(labels)) - (1 if -1 in model.labels_ else 0),
                "n_noise_": list(labels).count(-1),
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e


class SpectralClusteringWrapper(BaseClustering):
    """
    Wrapper for Spectral Clustering.
    Performs clustering via graph Laplacian and eigenvectors.
    """

    backend_name = "spectralclustering"
    frontend_name = "Spectral Clustering"

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        super().__init__(dataset_name, columns, preprocessing_params, **clustering_params)

    @staticmethod
    def get_default_params():
        return SpectralClustering().get_params()

    @staticmethod
    def get_sklearn_estimator_class():
        return SpectralClustering

    def run(self, data):
        """
        Apply Spectral Clustering and return labels and affinity matrix.
        """
        try:
            print(f"Running {self.frontend_name} with clustering_params: {self.clustering_params}")
            model = SpectralClustering(**self.clustering_params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(labels).items()}
            quality_metrics = self.compute_quality_metrics(data, labels)

            result = {
                "labels": [int(x) for x in labels.tolist()],
                "n_clusters": len(set(labels)),
                "affinity_matrix": model.affinity_matrix_.tolist() if hasattr(model, "affinity_matrix_") else None,
                "cluster_sizes": dict(cluster_sizes),
                "davies_bouldin_score": quality_metrics["davies_bouldin_score"],
                "calinski_harabasz_score": quality_metrics["calinski_harabasz_score"],
                "silhouette_score": quality_metrics["silhouette_score"],
            }

            return result

        except Exception as e:
            raise RuntimeError(f"{self.backend_name} failed: {e}") from e
