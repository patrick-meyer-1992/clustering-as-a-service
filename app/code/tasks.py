from celery_conn import celery
from datetime import datetime
import pytz

# Import all clustering wrapper classes
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering
from clustering.optics import OPTICSClustering
from clustering.meanshift import MeanShiftClustering
from clustering.affinitypropagation import AffinityPropagationClustering
from clustering.agglomerative import AgglomerativeClustering
from clustering.minibatchkmeans import MiniBatchKMeansClustering
from clustering.bisectingkmeans import BisectingKMeansClustering
from clustering.birch import BIRCHClustering
from clustering.spectral import SpectralClusteringClustering
from clustering.featureagglomeration import FeatureAgglomerationClustering
from clustering.spectralbiclustering import SpectralBiclusteringClustering
from clustering.spectralcoclustering import SpectralCoclusteringClustering

TIMEZONE = pytz.timezone("UTC")

# Map algorithm names to their corresponding classes
ALGORITHM_MAP = {
    "kmeans": KMeansClustering,
    "dbscan": DBSCANClustering,
    "optics": OPTICSClustering,
    "meanshift": MeanShiftClustering,
    "affinitypropagation": AffinityPropagationClustering,
    "agglomerative": AgglomerativeClustering,
    "minibatchkmeans": MiniBatchKMeansClustering,
    "bisectingkmeans": BisectingKMeansClustering,
    "birch": BIRCHClustering,
    "spectral": SpectralClusteringClustering,
    "featureagglomeration": FeatureAgglomerationClustering,
    "spectralbiclustering": SpectralBiclusteringClustering,
    "spectralcoclustering": SpectralCoclusteringClustering,
}

@celery.task(bind=True)
def run_clustering_job(self, dataset_name, columns, created_timestamp, clustering_algorithm, preprocess, user_id, **params):
    job_id = self.request.id
    started_timestamp = datetime.now(TIMEZONE).isoformat()

    algorithm_name = clustering_algorithm.lower()
    if algorithm_name not in ALGORITHM_MAP:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    ClusteringClass = ALGORITHM_MAP[algorithm_name]
    clustering = ClusteringClass(dataset_name, columns, **params)

    data = clustering.load_data()
    data = clustering.prepare_data(data, preprocess)
    result = clustering.run(data)
    clustering.save_results(result, job_id, created_timestamp, started_timestamp, user_id, original_data=data)

