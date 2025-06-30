import io
import os
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

# Environment variables for FastAPI connection
FASTAPI_HOST = os.getenv("FASTAPI_HOST")
FASTAPI_PORT = os.getenv("FASTAPI_PORT")
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL")


class BaseClustering(ABC):
    frontend_name = None
    backend_name = None

    def __init__(self, dataset_name, columns, **params):
        self.params = params
        self.dataset_name = dataset_name
        self.columns = columns
        self.name = "Base Clustering"  # Default name

    def load_data(self):
        try:
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{self.dataset_name}"
            print(f"Trying to load data from: {url}")  # Debug
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
            # Nur ausgewählte Spalten verwenden
            return df[self.columns].to_numpy()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    import numpy as np

    def validate_data(self, data):
        # Check for None
        if data is None:
            raise ValueError("Input data is None.")

        # Check shape attribute
        if not hasattr(data, "shape"):
            raise TypeError("Input data must be array-like with a shape attribute.")

        # Check empty
        if data.size == 0:
            raise ValueError("Input data is empty.")

        # Check 2D shape
        if len(data.shape) != 2:
            raise ValueError("Input data must be 2-dimensional (samples, features).")

        # Check if numeric
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("Input data must be numeric.")

    def validate_params(self):
        """
        Basic validation of input parameters.
        This does not replace sklearn's internal validation,
        but catches obvious issues early (e.g. wrong types, empty values).
        """
        for key, value in self.params.items():
            # Disallow empty strings
            if isinstance(value, str) and value.strip() == "":
                raise ValueError(f"Parameter '{key}' cannot be an empty string.")

            # Disallow None (except for known valid exceptions)
            if value is None and key not in ["preference", "init"]:
                raise ValueError(f"Parameter '{key}' cannot be None.")

            # Disallow negative numbers for common numerical parameters
            if isinstance(value, int | float) and key in ["n_clusters", "max_iter", "n_init"] and value <= 0:
                raise ValueError(f"Parameter '{key}' must be > 0.")

    def prepare_data(self, data, preprocess=True):
        # Apply preprocessing steps like scaling, normalization, and optional PCA
        X = data

        if not preprocess:
            return X

        df = pd.DataFrame(X)
        value_range = df.max() - df.min()

        # Determine scaler type from parameters or auto-select
        scaler_type = self.params.get("scaler", "auto")

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "maxabs":
            scaler = MaxAbsScaler()
        elif scaler_type == "auto":
            # Auto scaler selection based on feature range
            scaler = RobustScaler() if value_range.max() > 1000 or value_range.min() < 0.001 else StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        X_scaled = scaler.fit_transform(X)

        # Optional normalization
        use_normalization = self.params.get("use_normalization", False)
        norm_type = self.params.get("normalization_type", "l2")  # Options: 'l1', 'l2', 'max'

        if use_normalization:
            normalizer = Normalizer(norm=norm_type)
            X_scaled = normalizer.fit_transform(X_scaled)

        # Optional PCA dimensionality reduction
        use_pca = self.params.get("use_pca", False)
        pca_components = self.params.get("pca_components", 10)

        if use_pca and X_scaled.shape[1] > pca_components:
            pca = PCA(n_components=pca_components)
            X_scaled = pca.fit_transform(X_scaled)

        return X_scaled

    from abc import ABC

    @staticmethod
    @abstractmethod
    def get_default_params():
        """
        Must return a dictionary of default parameters.
        """
        pass

    @abstractmethod
    def run(self, data):
        # Abstract method to run the clustering algorithm
        pass

    def save_results(self, result, job_id, created_timestamp, started_timestamp, user_id):
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
                "clustering_algorithm": self.frontend_name,
                "params": self.params,
                "labels": labels,
                "additional_results": result,
                "user_id": user_id,
            }

            # Post the result to FastAPI backend
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/"
            print(f"Sending results to: {url}")  # Debug print
            response = requests.post(f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/result/", json=payload)

            if response.status_code != 200:
                print(f"Error saving results: {response.text}")
                return None

            return response.json()
        except Exception as e:
            print(f"Error in save_results: {str(e)}")
            return None
