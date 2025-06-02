from celery_conn import celery
from datetime import datetime
import pytz

# Import all clustering wrapper classes
from clustering.kmeans import KMeansWrapper
from clustering.dbscan import DBSCANWrapper
from clustering.optics import OPTICSWrapper
from clustering.meanshift import MeanShiftWrapper
from clustering.affinitypropagation import AffinityPropagationWrapper
from clustering.agglomerativeclustering import AgglomerativeClusteringWrapper
from clustering.minibatchkmeans import MiniBatchKMeansWrapper
from clustering.bisectingkmeans import BisectingKMeansWrapper
from clustering.birch import BIRCHWrapper
from clustering.spectralclustering import SpectralClusteringWrapper
from clustering.gaussianmixture import GaussianMixtureWrapper
from clustering.bayesiangaussianmixture import BayesianGaussianMixtureWrapper

TIMEZONE = pytz.timezone("UTC")

# Map algorithm names to their corresponding classes
ALGORITHM_MAP = {
    "kmeans": KMeansWrapper,
    "dbscan": DBSCANWrapper,
    "optics": OPTICSWrapper,
    "meanshift": MeanShiftWrapper,
    "affinitypropagation": AffinityPropagationWrapper,
    "agglomerative": AgglomerativeClusteringWrapper,
    "minibatchkmeans": MiniBatchKMeansWrapper,
    "bisectingkmeans": BisectingKMeansWrapper,
    "birch": BIRCHWrapper,
    "spectralclustering": SpectralClusteringWrapper,
    "gaussianmixture": GaussianMixtureWrapper,
    "bayesiangaussianmixture": BayesianGaussianMixtureWrapper,
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

