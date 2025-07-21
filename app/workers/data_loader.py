import io

import pandas as pd
import requests
from utils.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL
from utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_dataset(job_id, dataset_name, columns):
    url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{dataset_name}"
    logger.info(f"[AutoML][{job_id}] Fetching dataset from: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))
        logger.info(f"[AutoML][{job_id}] Raw dataset loaded, shape: {df.shape}")

        column_names = [col["name"] for col in columns]
        df = df[column_names]
    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Fetch dataset failed: {e}")
        raise

    logger.info(
        "[AutoML][%s] Dataset successfully filtered to columns %s, final shape: %s", job_id, column_names, df.shape
    )

    return df
