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
    print(f"Running clustering job for {dataset_name} with algorithm {clustering_algorithm}")
    started_timestamp = datetime.now(TIMEZONE).isoformat()

    job_id = self.request.id

    algorithm_name = clustering_algorithm.lower()
    if algorithm_name not in ALGORITHM_MAP:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    clustering_class = ALGORITHM_MAP[algorithm_name]
    clustering = clustering_class(dataset_name, columns, preprocessing_params, **clustering_params)

    # Clustering parameters validation
    # Temporarily removed because it throws false positive errors when n_clusters > 4
    # clustering.validate_params_sklearn()

    data = clustering.load_data()

    # Encoding nominal or ordinal columns
    df = clustering.encode_data(data)
    data = df.to_numpy()

    data = clustering.prepare_data(data, preprocess)
    result = clustering.run(data)

    clustering.save_results(result, job_id, created_timestamp, started_timestamp)
