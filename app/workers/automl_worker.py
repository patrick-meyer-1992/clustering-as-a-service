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

from sklearn.base import BaseEstimator

from .celery_conn import celery

# Environment variables (wie in deiner Base-Klasse)
fastapi_host = os.getenv("FASTAPI_HOST", "caas-fastapi")
fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
fastapi_protocol = os.getenv("FASTAPI_PROTOCOL", "http")

TIMEZONE = pytz.timezone("UTC")

###
# 
# cluster_alg_ls: This is the list of possible clustering algorithms to include within the search space.
# dim_reduction_alg_ls: This is the list of possible dimension reduction algorithms to include within the search space. 
#   Dimension reduction is performed before the clustering step. NullModel means no dimension reduction is done.
# optimizer: There are two options for this, "smac" or "random". "smac" does Bayesian Optimization 
#   using the SMAC library, while "random" just performs random search optimization.
# n_evaluations: number of iterations to run, generally the larger the better.
# cutoff_time: If evaluating a certain configuration takes longer than this value (in seconds), it will be terminated.
# preprocess_dict: This is important, AutoCluster.fit() uses this dictionary to preprocess the dataset. 
#   For instance, categorical columns will be one hot encoded, while ordinal columns will encoded as integers.
# evaluator: This is important, it tells AutoCluster.fit() how to evaluate a clustering result.
#     evaluator_ls: list of metric to include in a linear combination. 
#         Choices available are ["silhouetteScore", "daviesBouldinScore", "calinskiHarabaszScore"].
#     weights: how much weights to use for each metric in the linear combination.
#     clustering_num: A tuple is expected. If clustering result has n_clusters outside this specified 
#         range, float(inf) will be returned from evaluator.
#     min_proportion: The proportion of points in each cluster must be at least this value.
#     min_relative_proportion: The ratio of number points in the smallest cluster to the number of points in the 
#         largest cluster must be at least this value. By using 'default', min_relative_proportion will be set to  5 * min_proportion.
# warmstart: Whether or not to use warmstart, examples will be shown below on how to use this.
# verbose_level: Must be either 0, 1 or 2. The higher the number, the more logs/print statements are used. For normal usage we recommend using 1.
@celery.task(name="automl_worker.run_autocluster", bind=True)
def run_autocluster(self, *, dataset_name, columns):

    job_id = self.request.id

    print(f"[AutoML][{job_id}] Incoming automl_worker.run_autocluster: dataset_name={dataset_name}, columns={columns}")


    print("[DEBUG] pynisher has enforce_limits:", hasattr(pynisher, "enforce_limits"))

    try:
        # === Load dataset ===
        url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/{dataset_name}"
        
        print(f"[AutoML][{job_id}] Fetching dataset from {url}")
        

        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))
        df = df[columns]

        
        print(f"[AutoML][{job_id}] Dataset loaded successfully with shape {df.shape}")

        # === Configure AutoCluster ===
        
        
        clustering_algorithms = [
                'KMeans', 'GaussianMixture', 'Birch',
                'MiniBatchKMeans', 'AgglomerativeClustering', 'SpectralClustering']
        

        # ['NullModel']
        dim_redution_algorithms = [
                'TSNE', 'PCA', 'IncrementalPCA',
                'KernelPCA', 'FastICA', 'TruncatedSVD']


        optimizer = 'smac'

        n_evaluations = 50

        cutoff_time = 60

        evaluator_ls = ['silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore']


        print(f"[AutoML][{job_id}] Preparing fit parameters...")


        # === Fit-Konfiguration ===
        fit_params = {
            "df": df,
            "cluster_alg_ls": clustering_algorithms,
            "dim_reduction_alg_ls": dim_redution_algorithms,
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

        print(f"[AutoML][{job_id}] Starting AutoCluster fitting process with this config")
        print(f"[AutoML][{job_id}] {fit_params}")
        
        started_timestamp = datetime.now(TIMEZONE).isoformat()

        cluster = AutoCluster()
        result_dict = cluster.fit(**fit_params)


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
            "labels": {},
            "additional_results": sanitize_result_dict(result_dict)
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

