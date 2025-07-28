import io
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests
from pydantic import ValidationError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .preprocessing_params import PreProcessingParams
from .preprocessing_pipeline import PreprocessingPipeline

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

# Environment variables for FastAPI connection
FASTAPI_HOST = os.getenv("FASTAPI_HOST")
FASTAPI_PORT = os.getenv("FASTAPI_PORT")
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL")


DEFAULT_PREPROCESSING_PARAMS = {
    "scaler": "auto",
    "use_normalization": False,
    "normalization_type": "l2",
    "use_pca": False,
    "pca_components": 10,
    "transform_type": None,
    "imputation_strategy": None,
    "outlier_removal": None,  # None, "zscore", "iqr"
    "outlier_threshold": 3.0,  # only for zscore
    "feature_selection": None,  # None, "low_variance", "constant"
    "variance_threshold": 0.0,  # for low_variance
}


class BaseClustering(ABC):
    frontend_name = None
    backend_name = None

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        self.clustering_params = clustering_params
        self.dataset_name = dataset_name
        # TODO: @Ersel: Hier wird jetzt ein dict statt einer Liste erwartet.
        # Dieses dict kann für die Codierung der Spalten verwendet werden.
        self.columns: dict[str, str] = columns
        self.preprocessing_params = preprocessing_params
        self.name = "Base Clustering"  # Default name

        # Set default preprocessing parameters if not provided
        for k, v in self.get_default_params().items():
            if k not in self.clustering_params:
                self.clustering_params[k] = v

    def load_data(self):
        try:
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{self.dataset_name}"
            print(f"Trying to load data from: {url}")  # Debug
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
            # Nur ausgewählte Spalten verwenden
            return df[[col.get("name") for col in self.columns]]
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ordinal_cols = [col.get("name") for col in self.columns if col.get("type") == "ordinal"]
        nominal_cols = [col.get("name") for col in self.columns if col.get("type") == "nominal"]

        if ordinal_cols:
            encoder = OrdinalEncoder()
            df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])

        if nominal_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df[nominal_cols])
            new_cols = ohe.get_feature_names_out(nominal_cols)
            df_encoded = pd.DataFrame(encoded, columns=new_cols, index=df.index)

            df.drop(columns=nominal_cols, inplace=True)
            df = pd.concat([df, df_encoded], axis=1)

        return df

    def prepare_data(self, data, preprocess=True):
        if not preprocess:
            return data
        try:
            params_obj = PreProcessingParams(**(self.preprocessing_params or DEFAULT_PREPROCESSING_PARAMS))
        except ValidationError as ve:
            raise ValueError(f"Invalid preprocessing parameters: {ve}") from ve
        pipeline = PreprocessingPipeline(params_obj)
        return pipeline.fit_transform(data)

    @staticmethod
    @abstractmethod
    def get_default_params() -> dict:
        """
        Must return a dictionary of default parameters.
        """
        pass

    @abstractmethod
    def run(self, data):
        # Abstract method to run the clustering algorithm
        pass

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
                "clustering_algorithm": self.backend_name,
                "clustering_params": self.clustering_params,
                "preprocessing_params": self.preprocessing_params,
                "labels": labels,
                "additional_results": result,
            }

            payload = self._sanitize_inf(payload)

            # Post the result to FastAPI backend
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
            print(f"Sending results to: {url}")  # Debug print
            response = requests.post(
                f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/",
                json=payload,
            )

            if response.status_code != 200:
                print(f"Error saving results: {response.text}")
                return None

            return response.json()
        except Exception as e:
            print(f"Error in save_results: {str(e)}")
            return None

    def _sanitize_inf(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_inf(v) for v in obj]
        elif isinstance(obj, float):
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            if math.isnan(obj):
                return "nan"
            return obj
        elif isinstance(obj, np.ndarray):
            return self._sanitize_inf(obj.tolist())

        elif isinstance(obj, np.floating):
            if np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            if np.isnan(obj):
                return "nan"
            return float(obj)

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        else:
            return obj
