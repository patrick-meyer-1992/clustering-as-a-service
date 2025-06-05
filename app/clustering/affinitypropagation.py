from sklearn.cluster import AffinityPropagation
from .base_clustering import BaseClustering
import collections


class AffinityPropagationWrapper(BaseClustering):
    def __init__(self, dataset_name, columns, **params):
        super().__init__(dataset_name, columns)
        self.name = "AffinityPropagation"  # Exakt wie im Frontend
        self.params = {
            'damping': params.get('damping', 0.5),
            'max_iter': params.get('max_iter', 200),
            'convergence_iter': params.get('convergence_iter', 15),
            'copy': params.get('copy', True),
            'preference': params.get('preference', None),
            'affinity': params.get('affinity', 'euclidean'),
            'verbose': params.get('verbose', False)
        }

    def run(self, data):
        try:
            print(f"Running AffinityPropagation with params: {self.params}")
            model = AffinityPropagation(**self.params)
            model.fit(data)

            labels = model.labels_
            cluster_sizes = {int(k): v for k, v in collections.Counter(model.labels_).items()}

            result = {
                "labels": labels.tolist(),
                "cluster_centers_indices": model.cluster_centers_indices_.tolist(),
                "cluster_centers": model.cluster_centers_.tolist(),
                "n_iterations": model.n_iter_,
                "n_clusters": len(model.cluster_centers_indices_),
                "cluster_sizes": dict(cluster_sizes)
            }

            return result

        except Exception as e:
            print(f"Error in AffinityPropagation: {str(e)}")
            raise


