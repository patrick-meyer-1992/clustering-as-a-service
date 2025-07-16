import json
import subprocess
import os

from workers.celery_conn import celery


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, *, dataset_name, columns,
                    clustering_algorithms=None,
                    dim_reduction_algorithms=None,
                    n_evaluations=50,
                    cutoff_time=60,
                    evaluator_ls=None):

    job_id = self.request.id
    print(f"[AutoML][{job_id}] Delegating to subprocess...")
    print(f"[AutoML][{job_id}] {clustering_algorithms}")
    print(f"[AutoML][{job_id}] {dim_reduction_algorithms}")
    print(f"[AutoML][{job_id}] {evaluator_ls}")

    try:
        # Parameter an Subprozess übergeben
        script_path = os.path.join(os.path.dirname(__file__), "automl_worker.py")
        args = [
            "python", script_path,  # ruft dieselbe Datei erneut auf
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

        # Pass PYTHONPATH to subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"

        # Launch subprocess with logging
        result = subprocess.run(args, 
                                check=True, 
                                env=env, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, 
                                text=True
        )

        print(f"[AutoML][{job_id}] Subprocess output:\n{result.stdout}")
        

        return {"status": "submitted", "job_id": job_id}
    except subprocess.CalledProcessError as e:
        print(f"[AutoML][{job_id}] ERROR: {e}")
        return None
    