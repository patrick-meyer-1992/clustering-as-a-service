import os
import io
from datetime import datetime
import pynisher

import pandas as pd
import pytz
import requests


from autocluster import AutoCluster
from autocluster import get_evaluator
from autocluster import MetafeatureMapper

from .celery_conn import celery

# Environment variables (wie in deiner Base-Klasse)
fastapi_host = os.getenv("FASTAPI_HOST", "caas-fastapi")
fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
fastapi_protocol = os.getenv("FASTAPI_PROTOCOL", "http")

TIMEZONE = pytz.timezone("UTC")


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, *, dataset_name, columns):

    job_id = self.request.id

    print(f"[AutoML][{job_id}] Incoming automl_worker.run_autocluster: dataset_name={dataset_name}, columns={columns}")


    print("[DEBUG] pynisher has enforce_limits:", hasattr(pynisher, "enforce_limits"))

    try:
        # === Load dataset ===
        url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/{dataset_name}"
        url = "http://caas-fastapi:8000/dataset/iris.csv" # läd im moment nicht daher "fallback"
        
        print(f"[AutoML][{job_id}] Fetching dataset from {url}")
        
        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))
        df = df[columns]

        
        print(f"[AutoML][{job_id}] Dataset loaded successfully with shape {df.shape}")

        # === Configure AutoCluster ===
        evaluator_ls = ['silhouetteScore']
        print(f"[AutoML][{job_id}] Preparing fit parameters...")


        # === Fit-Konfiguration ===
        evaluator_ls = ['silhouetteScore']
        fit_params = {
            "df": df,
            "cluster_alg_ls": [
                'KMeans', 'GaussianMixture', 'Birch',
                'MiniBatchKMeans', 'AgglomerativeClustering', 'SpectralClustering'
            ],
            "dim_reduction_alg_ls": [
                'TSNE', 'PCA', 'IncrementalPCA',
                'KernelPCA', 'FastICA', 'TruncatedSVD'
            ],
            "optimizer": 'smac', # todo smac verwenden
            "n_evaluations": 40,
            "run_obj": 'quality',
            "seed": 27,
            "cutoff_time": 60,
            "preprocess_dict": {
                "numeric_cols": df.columns.tolist(),
                "categorical_cols": [],
                "ordinal_cols": [],
                "y_col": []
            },
            "evaluator": get_evaluator(
                evaluator_ls,
                weights=[],
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

        print(f"[AutoML][{job_id}] Starting AutoCluster fitting process...")
        
        started_timestamp = datetime.now(TIMEZONE).isoformat()

        print(f"[AutoML][{job_id}] Column names in df: {df.columns.tolist()}")
        print(f"[AutoML][{job_id}] Column names in numeric_cols: {fit_params['preprocess_dict']['numeric_cols']}")
        print(f"[AutoML][{job_id}] DataFrame head:\n{df.head()}")

        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)

        labels = result_dict.pop("labels", None)
        if labels is None:
            raise ValueError("Keine Labels im Ergebnis erhalten")

        print(f"[AutoML][{job_id}] AutoCluster finished successfully.")
        print(f"[AutoML][{job_id}] Result dictionary received from AutoCluster:")
        for key, value in result_dict.items():
            summary = str(value)
            if isinstance(value, (list, dict)) and len(summary) > 200:
                summary = summary[:200] + "... (truncated)"
            print(f"[AutoML][{job_id}]   - {key}: {summary}")

        payload = {
            "job_id": job_id,
            "dataset_name": dataset_name,
            "columns": columns,
            "created_timestamp": started_timestamp,
            "started_timestamp": started_timestamp,
            "finished_timestamp": datetime.now(TIMEZONE).isoformat(),
            "clustering_algorithm": "AutoCluster",
            "params": {},  # leer, da Autocluster intern konfiguriert
            "labels": labels,
            "additional_results": result_dict
        }

        result_url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/"
        print(f"[AutoML][{job_id}] Sending result to {result_url}")
        
        result_response = requests.post(result_url, json=payload, timeout=10)
        result_response.raise_for_status()

        print(f"[AutoML][{job_id}] Result successfully sent.")
        return result_response.json()

    except Exception as e:
        print(f"[AutoML][{job_id}] ERROR: {e}")
        return None
    
