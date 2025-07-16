from datetime import datetime
import requests
import numpy as np
from workers.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL, TIMEZONE

from sklearn.utils.validation import check_is_fitted


def send_results_to_backend(job_id, dataset_name, columns, started_timestamp, result_dict, cluster, df):
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

    predictions = cluster.predict(df)
    clustering_model = result_dict["clustering_model"]
    finished_timestamp = datetime.now(TIMEZONE).isoformat()

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

    url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
    print(f"[AutoML][{job_id}] Sending results to: {url}")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise Exception(f"[AutoML][{job_id}] Error saving results: {response.text}")
