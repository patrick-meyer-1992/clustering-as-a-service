import json
import subprocess
import os

from workers.celery_conn import celery
from workers.logger import setup_logger


logger = setup_logger(__name__)


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, *, dataset_name, columns,
                    clustering_algorithms=None,
                    dim_reduction_algorithms=None,
                    n_evaluations=50,
                    cutoff_time=60,
                    evaluator_ls=None):

    job_id = self.request.id

    logger.info(f"[AutoML][{job_id}] Task received. Delegating to subprocess.")
    logger.debug(f"[AutoML][{job_id}] clustering_algorithms: {clustering_algorithms}")
    logger.debug(f"[AutoML][{job_id}] dim_reduction_algorithms: {dim_reduction_algorithms}")
    logger.debug(f"[AutoML][{job_id}] evaluator_ls: {evaluator_ls}")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "automl_worker.py")

        args = [
            "python", script_path,
            job_id,
            dataset_name,
            json.dumps(columns),
            json.dumps({
                "clustering_algorithms": clustering_algorithms,
                "dim_reduction_algorithms": dim_reduction_algorithms,
                "n_evaluations": n_evaluations,
                "cutoff_time": cutoff_time,
                "evaluator_ls": evaluator_ls
            })
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"  

        logger.info(f"[AutoML][{job_id}] Starting subprocess: {' '.join(args)}")

        result = subprocess.run(
            args,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        logger.info(f"[AutoML][{job_id}] Subprocess finished successfully.")
        logger.debug(f"[AutoML][{job_id}] Subprocess output:\n{result.stdout}")

        return {"status": "submitted", "job_id": job_id}

    except subprocess.CalledProcessError as e:
        logger.error(f"[AutoML][{job_id}] Subprocess failed with return code {e.returncode}")
        logger.error(f"[AutoML][{job_id}] Output:\n{e.output}")
        return {"status": "error", "job_id": job_id, "error": str(e)}

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Unexpected error during subprocess execution")
        return {"status": "error", "job_id": job_id, "error": str(e)}