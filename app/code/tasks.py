from celery_conn import celery
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering

@celery.task
def run_clustering_job(dataset_url, columns, clustering_algorithm, user_id, params):
    if clustering_algorithm == "kmeans":
        clustering = KMeansClustering(dataset_url, **params)
    elif clustering_algorithm == "dbscan":
        clustering = DBSCANClustering(dataset_url, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {clustering_algorithm}")

    return clustering.run()
