import json
import os
import subprocess

from utils.logger import setup_logger

from workers.celery_conn import celery

logger = setup_logger(__name__)


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(
    self,
    *,
    dataset_name,
    columns,
    created_timestamp,
    clustering_algorithms=None,
    dim_reduction_algorithms=None,
    n_evaluations=50,
    cutoff_time=60,
    evaluator_ls=None,
):
    """
    Celery task to run an AutoML clustering process in a subprocess.

    This function prepares and starts a subprocess that executes the AutoML
    pipeline defined in `automl_worker.py`. All parameters are serialized and passed
    to the subprocess as command-line arguments.

    Parameters:
        dataset_name (str): The name of the dataset to be clustered.
        columns (list): A list of column metadata dictionaries to use in clustering.
        created_timestamp (str): Timestamp string indicating when the dataset was created.
        clustering_algorithms (list, optional): List of clustering algorithm names to evaluate.
        dim_reduction_algorithms (list, optional): List of dimensionality reduction methods to use.
        n_evaluations (int, optional): Number of configurations to evaluate. Defaults to 50.
        cutoff_time (int, optional): Maximum time (in seconds) for a single evaluation. Defaults to 60.
        evaluator_ls (list, optional): List of evaluator functions or scoring strategies.

    Returns:
        dict: A dictionary containing the task status and job ID. If an error occurs, includes an error message.

    Raises:
        subprocess.CalledProcessError: If the subprocess exits with a non-zero status.
        Exception: For any unexpected errors during subprocess invocation.
    """
        
    job_id = self.request.id

    logger.info(f"[AutoML][{job_id}] Task received. Delegating to subprocess.")
    logger.debug(f"[AutoML][{job_id}] clustering_algorithms: {clustering_algorithms}")
    logger.debug(f"[AutoML][{job_id}] dim_reduction_algorithms: {dim_reduction_algorithms}")
    logger.debug(f"[AutoML][{job_id}] evaluator_ls: {evaluator_ls}")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "automl_worker.py")

        args = [
            "python",
            script_path,
            job_id,
            dataset_name,
            json.dumps(columns),
            created_timestamp,
            json.dumps(
                {
                    "clustering_algorithms": clustering_algorithms,
                    "dim_reduction_algorithms": dim_reduction_algorithms,
                    "n_evaluations": n_evaluations,
                    "cutoff_time": cutoff_time,
                    "evaluator_ls": evaluator_ls,
                }
            ),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"

        completed_process = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True
        )

        if completed_process.returncode != 0:
            combined_output = (completed_process.stderr or "") + "\n" + (completed_process.stdout or "")
            # Finde die letzte sinnvolle Zeile (nicht leer)
            lines = [line.strip() for line in combined_output.strip().splitlines() if line.strip()]
            last_line = lines[-1] if lines else "Unknown error"

            logger.error(f"[AutoML][{job_id}] Subprocess failed: {last_line}")

            return {
                "status": "error",
                "job_id": job_id,
                "error": last_line,
            }

        return {"status": "SUCCESS", "job_id": job_id}

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Unexpected error during subprocess execution")
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e)
        }
