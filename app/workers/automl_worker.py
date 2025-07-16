import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import json
import traceback

from workers.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL, TIMEZONE
from workers.fit_config import prepare_fit_params
from workers.data_loader import fetch_dataset
from workers.result_handler import send_results_to_backend

from autocluster import AutoCluster


def run_autocluster_job(job_id, dataset_name, columns,
                        clustering_algorithms=None,
                        dim_reduction_algorithms=None,
                        n_evaluations=50,
                        cutoff_time=60,
                        evaluator_ls=None):

    print(f"[AutoML][{job_id}] Incoming automl_worker.run_autocluster")
    print(f"[AutoML][{job_id}] Params: {locals()}")

    try:
        df = fetch_dataset(job_id, dataset_name, columns)
        fit_params = prepare_fit_params(df, columns, clustering_algorithms, dim_reduction_algorithms, evaluator_ls, cutoff_time, n_evaluations)

        started_timestamp = datetime.now(TIMEZONE).isoformat()
        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)

        send_results_to_backend(job_id, dataset_name, columns, started_timestamp, result_dict, cluster, df)

        print(f"[AutoML][{job_id}] Result successfully sent.")
        return None

    except Exception as e:
        print(f"[AutoML][{job_id}] ERROR: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        job_id = sys.argv[1]
        dataset_name = sys.argv[2]
        columns = json.loads(sys.argv[3])
        optional_params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}

        print("---- CLI ARGUMENTS ----")
        print("job_id:", job_id)
        print("dataset_name:", dataset_name)
        print("columns:", columns)
        print("optional_params:", optional_params)
        print("------------------------")

        run_autocluster_job(
            job_id=job_id,
            dataset_name=dataset_name,
            columns=columns,
            **optional_params
        )

    except Exception as e:
        print("ERROR during __main__ execution:")
        traceback.print_exc()
