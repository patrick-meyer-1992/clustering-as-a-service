import requests
import json
from time import sleep

FASTAPI_URL = f"http://api.staging.caas.local:80"
# FASTAPI_URL = f"http://localhost:7001"

columns = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width",
]

data = {
    "dataset_name": "iris.csv",
    "columns": columns,
    "clustering_algorithm": "kmeans",
    "preprocess": "true",
    "clustering_params": {},
    "preprocessing_params": {}
}

iterations = 5000
sleep_time = 0.1
for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}")
    res = requests.post(f"{FASTAPI_URL}/cluster/", json=data)
    print(res.text)
    sleep(sleep_time)
