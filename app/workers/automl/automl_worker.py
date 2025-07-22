import json
import os
import sys
from datetime import datetime

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocluster import AutoCluster
from utils.config import TIMEZONE
from utils.logger import setup_logger

from workers.automl.data_loader import fetch_dataset
from workers.automl.fit_config import prepare_fit_params
from workers.automl.result_handler import send_results_to_backend

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
    clustering_num=None,
):
    """
    Executes an AutoML clustering job synchronously.

    This function fetches the dataset, prepares fit parameters, runs the
    AutoCluster pipeline, and sends the resulting metadata and metrics
    back to the backend service.

    Parameters:
        job_id (str): Unique identifier for the clustering job.
        dataset_name (str): The name of the dataset to be clustered.
        columns (list): List of column definitions (e.g., name, type).
        created_timestamp (str): Timestamp indicating dataset creation time.
        clustering_algorithms (list, optional): Algorithms to be evaluated.
        dim_reduction_algorithms (list, optional): Dimensionality reduction methods.
        n_evaluations (int, optional): Number of configurations to evaluate. Default is 50.
        cutoff_time (int, optional): Max time per evaluation in seconds. Default is 60.
        evaluator_ls (list, optional): Evaluation metrics or scoring functions.

    Returns:
        None: The function does not return a value; it triggers side-effects like logging and result posting.

    Raises:
        Exception: Logs and swallows any exceptions that occur during execution.
    """

    logger.info(
        f"[AutoML][{job_id}] Incoming job | dataset_name={dataset_name}, "
        f"n_evaluations={n_evaluations}, cutoff_time={cutoff_time}"
    )

    try:
        logger.debug(f"[AutoML][{job_id}] Fetching dataset")
        df = fetch_dataset(job_id, dataset_name, columns)

        logger.debug(f"[AutoML][{job_id}] Preparing fit params")
        fit_params = prepare_fit_params(
            df, columns, clustering_algorithms, dim_reduction_algorithms, evaluator_ls, cutoff_time, n_evaluations, clustering_num
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
        raise e


if __name__ == "__main__":
    """
    Script entry point to execute an AutoML clustering job from the command line.

    Expects the following command-line arguments:
        1. job_id (str)
        2. dataset_name (str)
        3. columns (JSON-encoded list)
        4. created_timestamp (str)
        5. optional_params (JSON-encoded dict; optional)

    Example usage:
        python automl_worker.py <job_id> <dataset_name> '<columns_json>' <created_timestamp> '<optional_params_json>'
    """
    
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

        result = run_autocluster_job(
            job_id=job_id,
            dataset_name=dataset_name,
            columns=columns,
            created_timestamp=created_timestamp,
            **optional_params,
        )

        if result:
            print(f"[AutoML][{job_id}] Clustering failed: {result}")
            sys.exit(1)
        else:
            print(f"[AutoML][{job_id}] Clustering completed successfully.")

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Exception in __main__ block {e}")
        sys.exit(1)
