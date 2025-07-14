from datetime import datetime

import pandas as pd
import pytz
from clustering import wrappers

from .celery_conn import celery

TIMEZONE = pytz.timezone("UTC")

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
    try:
        print(f"Running clustering job for {dataset_name} with algorithm {clustering_algorithm}")
        started_timestamp = datetime.now(TIMEZONE).isoformat()

        job_id = self.request.id

        algorithm_name = clustering_algorithm.lower()
        if algorithm_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        clustering_class = ALGORITHM_MAP[algorithm_name]
        clustering = clustering_class(dataset_name, columns, preprocessing_params, **clustering_params)

        # Clustering parameters validation
        clustering.validate_params_sklearn()

        data = clustering.load_data()

        # Encoding nominal or ordinal columns
        df = pd.DataFrame(data)
        df = clustering.encode_data(df)
        data = df.to_numpy()

        data = clustering.prepare_data(data, preprocess)
        result = clustering.run(data)

        # Debugging: Logge die erweiterten Ergebnisse
        print(f"Result with X and columns for job_id {job_id}: {result}")

        clustering.save_results(result, job_id, created_timestamp, started_timestamp)

    except Exception as e:
        print(f"Error in clustering job: {str(e)}")
        return {"status": "error", "message": str(e)}
