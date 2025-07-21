import requests
import json
from time import sleep

# FASTAPI_URL = f"http://api.staging.caas.local:80"
FASTAPI_URL = f"http://localhost:7001"

data = {
    "clustering_algorithm": "kmeans",
    "clustering_params": {"n_clusters": 3},
    "columns": [
        {"name": "sepal.length", "type": "numeric"},
        {"name": "sepal.width", "type": "numeric"},
    ],
    "dataset_name": "iris.csv",
    "preprocess": "true",
    "preprocessing_params": {
        "feature_selection": "low_variance",
        "imputation_strategy": "mean",
        "normalization_type": "l2",
        "outlier_removal": "zscore",
        "outlier_threshold": 3,
        "pca_components": 10,
        "scaler": "auto",
        "transform_type": "quantile",
        "use_normalization": "false",
        "use_pca": "false",
        "variance_threshold": 0,
    },
}

iterations = 10
sleep_time = 1
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")
    res = requests.post(f"{FASTAPI_URL}/job/", json=data)
    print(res.text)
    sleep(sleep_time)
