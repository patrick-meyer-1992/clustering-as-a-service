from .celery_conn import celery
from datetime import datetime
import pytz
from clustering import wrappers

TIMEZONE = pytz.timezone("UTC")

ALGORITHM_MAP = {getattr(wrappers, algo).backend_name: getattr(wrappers, algo) for algo in dir(wrappers) if algo.endswith('Wrapper')}

@celery.task(bind=True)
def run_clustering_job(self, dataset_name, columns, created_timestamp, clustering_algorithm, preprocess, user_id, **params):
    try:
        print(f"Running clustering job for {dataset_name} with algorithm {clustering_algorithm}")
        started_timestamp = datetime.now(TIMEZONE).isoformat()

        job_id = self.request.id

        algorithm_name = clustering_algorithm.lower()
        if algorithm_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        clustering_class = ALGORITHM_MAP[algorithm_name]
        clustering = clustering_class(dataset_name, columns, **params)

        data = clustering.load_data()
        data = clustering.prepare_data(data, preprocess)
        result = clustering.run(data)
        clustering.save_results(result, job_id, created_timestamp, started_timestamp, user_id)

    except Exception as e:
        print(f"Error in clustering job: {str(e)}")
        return {"status": "error", "message": str(e)}
