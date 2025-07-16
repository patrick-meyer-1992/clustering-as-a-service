import requests
import io
import pandas as pd

from workers.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL, TIMEZONE


def fetch_dataset(job_id, dataset_name, columns):
    url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{dataset_name}"
    print(f"[AutoML][{job_id}] Fetching dataset from {url}")

    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))
    column_names = [col["name"] for col in columns]
    df = df[column_names]

    print(f"[AutoML][{job_id}] Dataset loaded successfully with shape {df.shape}")
    return df