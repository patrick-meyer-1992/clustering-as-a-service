import json
import os
import subprocess
from utils.logger import setup_logger
from workers.celery_conn import celery

logger = setup_logger(__name__)


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, **kwargs):
    job_id = self.request.id
    return run_autocluster_logic(job_id=job_id, **kwargs)


def run_autocluster_logic(
    *,
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
    min_proportion=None,
    min_relative_proportion=None
):
    logger.info(f"[AutoML][{job_id}] Task received. Delegating to subprocess.")
    logger.debug(f"[AutoML][{job_id}] clustering_algorithms: {clustering_algorithms}")
    logger.debug(f"[AutoML][{job_id}] dim_reduction_algorithms: {dim_reduction_algorithms}")
    logger.debug(f"[AutoML][{job_id}] evaluator_ls: {evaluator_ls}")

    try:
        args = build_subprocess_args(
            job_id,
            dataset_name,
            columns,
            created_timestamp,
            clustering_algorithms,
            dim_reduction_algorithms,
            n_evaluations,
            cutoff_time,
            evaluator_ls,
            clustering_num,
            min_proportion,
            min_relative_proportion
        )
        env = build_environment()

        return run_subprocess(job_id, args, env)

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Unexpected error during subprocess execution")
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e)
        }


def build_subprocess_args(
    job_id,
    dataset_name,
    columns,
    created_timestamp,
    clustering_algorithms,
    dim_reduction_algorithms,
    n_evaluations,
    cutoff_time,
    evaluator_ls,
    clustering_num,
    min_proportion,
    min_relative_proportion
):
    script_path = os.path.join(os.path.dirname(__file__), "automl_worker.py")

    config_dict = {
        "clustering_algorithms": clustering_algorithms,
        "dim_reduction_algorithms": dim_reduction_algorithms,
        "n_evaluations": n_evaluations,
        "cutoff_time": cutoff_time,
        "evaluator_ls": evaluator_ls,
        "clustering_num": clustering_num,
        "min_proportion": min_proportion,
        "min_relative_proportion": min_relative_proportion
    }

    args = [
        "python",
        script_path,
        job_id,
        dataset_name,
        json.dumps(columns),
        created_timestamp,
        json.dumps(config_dict),
    ]
    return args


def build_environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    return env


def run_subprocess(job_id, args, env):
    process = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    error_message = None

    with process.stdout:
        for line in process.stdout:
            line = line.strip()
            logger.info(f"[AutoML][{job_id}] {line}")
            if "Traceback" in line or "Error" in line or "Exception" in line:
                error_message = line

    return_code = process.wait()

    if return_code != 0:
        logger.error(f"[AutoML][{job_id}] Subprocess failed with return code {return_code}")
        return {
            "status": "error",
            "job_id": job_id,
            "error": error_message or f"Subprocess failed with return code {return_code}"
        }

    return {"status": "SUCCESS", "job_id": job_id}


