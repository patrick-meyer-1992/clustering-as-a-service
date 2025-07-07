import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
from datetime import datetime
import pynisher
import subprocess
import json
import pandas as pd
import pytz
import requests
import numpy as np

from collections import Counter

from autocluster import AutoCluster
from autocluster import get_evaluator
from autocluster import MetafeatureMapper

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from workers.celery_conn import celery

# Environment variables (wie in deiner Base-Klasse)
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "caas-fastapi")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL", "http")

TIMEZONE = pytz.timezone("UTC")


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

    # Parameter an Subprozess übergeben
    script_path = os.path.abspath(__file__)
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

    try:
        subprocess.run(args, check=True)
        return {"status": "submitted", "job_id": job_id}
    except subprocess.CalledProcessError as e:
        print(f"[AutoML][{job_id}] ERROR: {e}")
        return None
    
    

def is_json_serializable(obj):
    try:
        import json
        json.dumps(obj)
        return True
    except:
        return False

def sanitize_result_dict(result_dict):
    cleaned = {}
    for k, v in result_dict.items():
        if k in {"smac_obj", "random_optimizer_obj", "clustering_model", "dim_reduction_model", "scaler"}:
            continue
        elif k == "optimal_cfg":
            cleaned[k] = str(v)
        elif isinstance(v, BaseEstimator):
            cleaned[k] = str(v)
        elif hasattr(v, "tolist"):
            cleaned[k] = v.tolist()
        elif is_json_serializable(v):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def save_results(self, result, job_id, created_timestamp, started_timestamp):
    """
    Save clustering results to FastAPI backend
    """
    try:
        print(f"Saving results for job_id: {job_id}")  # Debug print
        # Pop labels from result dictionary
        labels = result.pop("labels")
        
        payload = {
            "job_id": job_id,  # Hier verwenden wir den übergebenen job_id Parameter
            "dataset_name": self.dataset_name,
            "columns": self.columns,
            "created_timestamp": created_timestamp,
            "started_timestamp": started_timestamp,
            "finished_timestamp": datetime.now(TIMEZONE).isoformat(),
            "clustering_algorithm": self.frontend_name,
            "clustering_params": self.clustering_params,
            "preprocessing_params": self.preprocessing_params,
            "labels": labels,
            "additional_results": result,
        }

        # Post the result to FastAPI backend
        url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
        print(f"Sending results to: {url}")  # Debug print
        response = requests.post(f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/", json=payload)

        if response.status_code != 200:
            print(f"Error saving results: {response.text}")
            return None
    except Exception as e:
        print(f"Error in save_results: {str(e)}")
        return None
            

def run_autocluster_job(job_id, dataset_name, columns,
                        clustering_algorithms=None,
                        dim_reduction_algorithms=None,
                        n_evaluations=50,
                        cutoff_time=60,
                        evaluator_ls=None):

    print(f"[AutoML][{job_id}] Incoming automl_worker.run_autocluster: dataset_name={dataset_name}, columns={columns}, n_evaluations={n_evaluations}, cutoff_time={cutoff_time}")
    print(f"[AutoML][{job_id}] clustering_algorithms: {clustering_algorithms}")
    print(f"[AutoML][{job_id}] dim_reduction_algorithms: {dim_reduction_algorithms}")
    print(f"[AutoML][{job_id}] evaluator_ls: {evaluator_ls}")

    print("[DEBUG] pynisher has enforce_limits:", hasattr(pynisher, "enforce_limits"))

    try:
        # === Load dataset ===
        url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{dataset_name}"
        
        print(f"[AutoML][{job_id}] Fetching dataset from {url}")
        

        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))
        df = df[columns]

        
        print(f"[AutoML][{job_id}] Dataset loaded successfully with shape {df.shape}")

        # === Configure AutoCluster ===
        if not dim_reduction_algorithms:
            dim_reduction_algorithms = [
                'NullModel'
            ]

        if not evaluator_ls:
            evaluator_ls = ['silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore']


        optimizer = 'smac'



        print(f"[AutoML][{job_id}] Preparing fit parameters...")
        print(f"[AutoML][{job_id}] Final algorithms: {clustering_algorithms}")
        print(f"[AutoML][{job_id}] Final dim_red: {dim_reduction_algorithms}")
        print(f"[AutoML][{job_id}] Evaluators: {evaluator_ls}")

        # === Fit-Konfiguration ===
        fit_params = {
            "df": df,
            "cluster_alg_ls": clustering_algorithms,
            "dim_reduction_alg_ls": dim_reduction_algorithms,
            "optimizer": optimizer, 
            "n_evaluations": n_evaluations,
            "run_obj": 'quality',
            "seed": 27,
            "cutoff_time": cutoff_time,
            "preprocess_dict": {
                "numeric_cols": df.columns.tolist(),
                "categorical_cols": [],
                "ordinal_cols": [],
                "y_col": []
            },
            "evaluator": get_evaluator(
                evaluator_ls,
                weights=[1, 1, 1],
                clustering_num=None,
                min_proportion=.01
            ),
            "n_folds": 3,
            "warmstart": False,
            "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
            "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
            "categorical_metafeatures": [],
            "verbose_level": 1,
        }

        print(f"[AutoML][{job_id}] Starting AutoCluster fitting process with this config")
        print(f"[AutoML][{job_id}] {fit_params}")
        
        started_timestamp = datetime.now(TIMEZONE).isoformat()

        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)


        # Path mitgeben und graph speichern
        predictions = cluster.predict(df)
        

        print(f"[AutoML][{job_id}] Sending result")
        # save_results(self, result, job_id, created_timestamp, started_timestamp):
        try:
            
            def sanitize_result_dict(result_dict):
                def safe_str_or_list(value):
                    if hasattr(value, "tolist"):
                        return value.tolist()
                    elif isinstance(value, (dict, list, int, float, str, type(None))):
                        return value
                    return str(value)

                return {
                    "optimal_cfg": str(result_dict.get("optimal_cfg")),
                    "metafeatures_used": result_dict.get("metafeatures_used", []),
                    "metafeatures": safe_str_or_list(result_dict.get("metafeatures")),
                }

            def extract_params_from_model(model):
                try:
                    check_is_fitted(model)
                    return model.get_params()
                except Exception:
                    return {}

            # === Prediction ===
            predictions = cluster.predict(df)

            # === Werte extrahieren ===
            optimal_cfg = result_dict["optimal_cfg"]
            clustering_model = result_dict["clustering_model"]
            scaler = result_dict["scaler"]

            # === Timestamp ===
            finished_timestamp = datetime.now(TIMEZONE).isoformat()

            # === Payload für POST ===
            payload = {
                "job_id": job_id,
                "dataset_name": dataset_name,
                "columns": columns,
                "created_timestamp": started_timestamp,
                "started_timestamp": started_timestamp,
                "finished_timestamp": finished_timestamp,
                "clustering_algorithm": getattr(clustering_model, "__class__", type("Unknown", (), {})).__name__.lower(),
                "clustering_params": extract_params_from_model(clustering_model),
                "preprocessing_params": {"scaler": "standard"},
                "labels": predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions),
                "additional_results": sanitize_result_dict(result_dict)
            }

            # Post the result to FastAPI backend
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
            print(f"Sending results to: {url}")  # Debug print
            response = requests.post(f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/", json=payload)

            if response.status_code != 200:
                print(f"Error saving results: {response.text}")
                return None

        except Exception as e:
            print(f"Error in save_results: {str(e)}")
            return None
        
        print(f"[AutoML][{job_id}] Result successfully sent.")
        return None

    except Exception as e:
        print(f"[AutoML][{job_id}] ERROR: {e}")
        return None


if __name__ == "__main__":
    import sys
    import json

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

