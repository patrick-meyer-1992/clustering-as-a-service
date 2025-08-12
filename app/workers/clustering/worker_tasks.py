from datetime import datetime

from clustering import wrappers
from utils.config import TIMEZONE
from utils.logger import setup_logger
from workers.celery_conn import celery

logger = setup_logger(__name__)

ALGORITHM_MAP = {
    getattr(wrappers, algo).backend_name: getattr(wrappers, algo) for algo in dir(wrappers) if algo.endswith("Wrapper")
}


@celery.task(bind=True)
def run_clustering_job(
    self,
    dataset_name,
    columns,
    created_timestamp,
    clustering_algorithm,
    preprocess,
    preprocessing_params,
    **clustering_params,
):
    job_id = self.request.id
    logger.info(f"[{job_id}] Starting clustering job: {clustering_algorithm} on dataset: {dataset_name}")

    started_timestamp = datetime.now(TIMEZONE).isoformat()

    try:
        algorithm_name = clustering_algorithm.lower()
        if algorithm_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        clustering_class = ALGORITHM_MAP[algorithm_name]
        clustering = clustering_class(dataset_name, columns, preprocessing_params, **clustering_params)

        df = clustering.load_data()
        df = clustering.encode_data(df)
        data = df.to_numpy()

        data = clustering.prepare_data(data, preprocess)

        result = clustering.run(data)

        response = clustering.save_results(result, job_id, created_timestamp, started_timestamp)

        logger.info(f"[{job_id}] Clustering job completed successfully.")
        return response

    except Exception as e:
        logger.error(f"[{job_id}] Clustering job failed: {str(e)}", exc_info=True)
        raise
