import logging
from datetime import datetime

import numpy as np
import requests
from sklearn.utils.validation import check_is_fitted

from utils.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL, TIMEZONE

logger = logging.getLogger(__name__)


def send_results_to_backend(
    job_id, dataset_name, columns, created_timestamp, started_timestamp, result_dict, cluster, df
):
    """
    Sends the clustering results and metadata to the FastAPI backend for persistence.

    This function collects relevant results from the AutoML clustering run,
    including model parameters, predictions, metafeatures, and timestamps.
    It then formats and posts this information as JSON to the backend API.

    Parameters:
        job_id (str): Unique identifier for the current AutoML task.
        dataset_name (str): The name of the dataset used in the clustering task.
        columns (list): List of column metadata dicts used in the run.
        created_timestamp (str): Original timestamp of dataset creation.
        started_timestamp (str): Timestamp when clustering was started.
        result_dict (dict): Dictionary containing clustering results and metadata.
        cluster (object): AutoCluster instance used for fitting and prediction.
        df (pandas.DataFrame): The dataset used for clustering.

    Returns:
        None

    Raises:
        Exception: If the HTTP POST request fails or any critical error occurs during transmission.

    Notes:
        - Uses internal helper `sanitize_result_dict()` to prepare serializable result entries.
        - Uses `extract_params_from_model()` to retrieve model hyperparameters, if available.
        - Automatically adds timestamps and converts NumPy arrays to lists for JSON compatibility.
        - Sends data to the `/result/` endpoint of the FastAPI backend.
    """

    logger.info(f"[AutoML][{job_id}] Preparing to send results for dataset '{dataset_name}'")

    def sanitize_result_dict(result_dict):
        def safe_str_or_list(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            elif isinstance(value, (dict | list | int | float | str | type(None))):
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
        except Exception as e:
            logger.exception(f"[AutoML][{job_id}] Model not fitted or param extraction failed: {e}")
            return {}

    try:
        predictions = cluster.predict(df, save_plot=False)
        clustering_model = result_dict["clustering_model"]
        finished_timestamp = datetime.now(TIMEZONE).isoformat()

        payload = {
            "job_id": job_id,
            "dataset_name": dataset_name,
            "columns": columns,
            "created_timestamp": created_timestamp,
            "started_timestamp": started_timestamp,
            "finished_timestamp": finished_timestamp,
            "clustering_algorithm": "AutoML",
            "clustering_params": extract_params_from_model(clustering_model),
            "preprocessing_params": {"scaler": "standard"},
            "labels": predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions),
            "additional_results": sanitize_result_dict(result_dict),
        }

        url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
        logger.info(f"[AutoML][{job_id}] Sending results to backend at {url}")
        logger.debug(f"[AutoML][{job_id}] Payload: {payload}")

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            logger.info(f"[AutoML][{job_id}] Results successfully sent.")
        else:
            logger.error(f"[AutoML][{job_id}] Failed to send results: {response.status_code} - {response.text}")
            raise Exception(f"Error saving results: {response.text}")

    except Exception as e:
        logger.exception(f"[AutoML][{job_id}] Exception during result transmission: {e}")
