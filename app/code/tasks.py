from celery_conn import celery
from datetime import datetime
import pytz
import importlib

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

@celery.task
def run_clustering_job(dataset_name, columns, created_timestamp, module_name, class_name, preprocess, user_id, **params):
    try:
        print(f"Running clustering job for {dataset_name} with {class_name}")
        
        # Import clustering module
        module = importlib.import_module(f"clustering.{module_name}")
        clustering_class = getattr(module, class_name)
        
        # Create clustering instance
        clustering = clustering_class(dataset_name, columns, **params)
        
        # Load and prepare data
        data = clustering.load_data()
        prepared_data = clustering.prepare_data(data, preprocess)
        
        # Run clustering
        started_timestamp = datetime.now(TIMEZONE).isoformat()
        result = clustering.run(prepared_data)
        
        # Save results using the task ID as job_id
        task_id = run_clustering_job.request.id
        clustering.save_results(
            result,
            task_id,  # Hier verwenden wir die Task ID als job_id
            created_timestamp,
            started_timestamp,
            user_id,
            prepared_data
        )
        
        return {"status": "success", "job_id": task_id}
        
    except Exception as e:
        print(f"Error in clustering job: {str(e)}")
        return {"status": "error", "message": str(e)}

