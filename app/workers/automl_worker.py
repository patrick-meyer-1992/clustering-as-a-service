import json
import os
import sys
from datetime import datetime

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocluster import AutoCluster

from workers.config import TIMEZONE
from workers.data_loader import fetch_dataset
from workers.fit_config import prepare_fit_params
from workers.logger import setup_logger
from workers.result_handler import send_results_to_backend

logger = setup_logger(__name__)


def run_autocluster_job(
    job_id,
    dataset_name,
    columns,
    created_timestamp,
    clustering_algorithms=None,
    dim_reduction_algorithms=None,
    n_evaluations=50,
    cutoff_time=60,
    evaluator_ls=None,
):
    logger.info(
        f"[AutoML][{job_id}] Incoming job | dataset_name={dataset_name}, "
        f"n_evaluations={n_evaluations}, cutoff_time={cutoff_time}"
    )

    try:
        logger.debug(f"[AutoML][{job_id}] Fetching dataset")
        df = fetch_dataset(job_id, dataset_name, columns)

        logger.debug(f"[AutoML][{job_id}] Preparing fit params")
        fit_params = prepare_fit_params(
            df, columns, clustering_algorithms, dim_reduction_algorithms, evaluator_ls, cutoff_time, n_evaluations
        )

        started_timestamp = datetime.now(TIMEZONE).isoformat()

        logger.debug(f"[AutoML][{job_id}] Running autocluster")
        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)

        logger.debug(f"[AutoML][{job_id}] Sending results to backend")
        send_results_to_backend(
            job_id, dataset_name, columns, created_timestamp, started_timestamp, result_dict, cluster, df
        )

        logger.info(f"[AutoML][{job_id}] Result successfully sent.")
        return None

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Unhandled error during AutoML job {e}")
        return None


if __name__ == "__main__":
    try:
        job_id = sys.argv[1]
        dataset_name = sys.argv[2]
        columns = json.loads(sys.argv[3])
        created_timestamp = sys.argv[4]
        optional_params = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

        logger.info(
            f"[AutoML][{job_id}] Subprocess started | "
            f"dataset_name={dataset_name}, "
            f"columns={columns}, "
            f"created_timestamp={created_timestamp}, "
            f"optional_params={optional_params}"
        )

        run_autocluster_job(job_id=job_id, dataset_name=dataset_name, columns=columns, 
                            created_timestamp=created_timestamp, **optional_params)

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Exception in __main__ block {e}")
