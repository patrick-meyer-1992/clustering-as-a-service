import io
import os
from abc import ABC, abstractmethod
from datetime import datetime

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
fastapi_host = os.getenv("FASTPI_HOST")
fastapi_port = os.getenv("FASTAPI_PORT")
fastapi_protocol = os.getenv("FASTAPI_PROTOCOL")


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
            url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/{self.dataset_name}"
            print(f"Trying to load data from: {url}")  # Debug
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
            # Nur ausgewählte Spalten verwenden
            return df[self.columns].to_numpy()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

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
            if value_range.max() > 1000 or value_range.min() < 1e-3:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
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

    from abc import ABC, abstractmethod
    
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
            url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/"
            print(f"Sending results to: {url}")  # Debug print
            response = requests.post(
                f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/", json=payload
            )

            if response.status_code != 200:
                print(f"Error saving results: {response.text}")
                return None

            return response.json()
        except Exception as e:
            print(f"Error in save_results: {str(e)}")
            return None
