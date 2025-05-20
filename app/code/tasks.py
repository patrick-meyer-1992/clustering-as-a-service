from celery_conn import celery
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering
from datetime import datetime
import pytz

TIMEZONE = pytz.timezone("UTC")

@celery.task(bind=True)
def run_clustering_job(self, dataset_name, columns, created_timestamp, clustering_algorithm, preprocess, user_id, **params):
    job_id = self.request.id  # Access the current task's id
    started_timestamp = datetime.now(TIMEZONE).isoformat()
    if clustering_algorithm == "kmeans":
        clustering = KMeansClustering(dataset_name, columns, **params)
    elif clustering_algorithm == "dbscan":
        clustering = DBSCANClustering(dataset_name, columns, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {clustering_algorithm}")

    data = clustering.load_data()
    data = clustering.prepare_data(data, preprocess)
    result = clustering.run(data)
    clustering.save_results(result, job_id, created_timestamp, started_timestamp, user_id)
