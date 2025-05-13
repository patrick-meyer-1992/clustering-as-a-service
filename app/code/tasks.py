from celery_conn import celery
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering

@celery.task
def run_clustering_task(data, algorithm, params):
    if algorithm == "kmeans":
        clustering = KMeansClustering(data, **params)
    elif algorithm == "dbscan":
        clustering = DBSCANClustering(data, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return clustering.run()
