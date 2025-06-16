import os
import io
from datetime import datetime

import pandas as pd
import pytz
import requests

from autocluster import AutoCluster
from autocluster import get_evaluator
from autocluster import MetafeatureMapper

from .celery_conn import celery

# Environment variables (wie in deiner Base-Klasse)
fastapi_host = os.getenv("FASTPI_HOST")
fastapi_port = os.getenv("FASTAPI_PORT")
fastapi_protocol = os.getenv("FASTAPI_PROTOCOL")

TIMEZONE = pytz.timezone("UTC")


@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, *, dataset_name, columns, user_id):
    print(f"[AutoML] Eingehende Parameter: dataset_name={dataset_name}, columns={columns}, user_id={user_id}")

    job_id = self.request.id
    try:
        # === Daten laden ===
        url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/{dataset_name}"
        print(f"[AutoML] Lade Daten von {url}")
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
        df = df[columns]

        # todo welche Parameter sollen einstellbar sein?
        # algorithmen



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
            "optimizer": 'smac',
            "n_evaluations": 40,
            "run_obj": 'quality',
            "seed": 27,
            "cutoff_time": 60,
            "preprocess_dict": {
                "numeric_cols": list(range(df.shape[1])), 
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
            "warmstart": True,
            "warmstart_datasets_dir": 'experiments/metaknowledge/benchmark_silhouette',
            "warmstart_metafeatures_table_path": 'experiments/metaknowledge/benchmark_silhouette_metafeatures_table.csv',
            "warmstart_n_neighbors": 10,
            "warmstart_top_n": 3,
            "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
            "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
            "categorical_metafeatures": [],
            "verbose_level": 1,
        }

        started_timestamp = datetime.now(TIMEZONE).isoformat()
        print("[AutoML] Starte AutoCluster")
        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)

        labels = result_dict.pop("labels", None)
        if labels is None:
            raise ValueError("Keine Labels im Ergebnis erhalten")

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
            "additional_results": result_dict,
            "user_id": user_id,
        }

        result_url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/"
        print(f"[AutoML] Sende Ergebnisse an {result_url}")
        result_response = requests.post(result_url, json=payload)

        if result_response.status_code != 200:
            print(f"[AutoML] Fehler beim Senden der Ergebnisse: {result_response.text}")
            return None

        print("[AutoML] Erfolg")
        return result_response.json()

    except Exception as e:
        print(f"[AutoML] Fehler: {e}")
        return None
