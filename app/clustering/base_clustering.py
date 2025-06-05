from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import requests
import os
import pandas as pd
import io
from datetime import datetime
import pytz

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

# Environment variables for FastAPI connection
fastapi_host = os.getenv('FASTPI_HOST')
fastapi_port = os.getenv('FASTAPI_PORT')
fastapi_protocol = os.getenv('FASTAPI_PROTOCOL')

class BaseClustering(ABC):
    def __init__(self, dataset_name, columns, **params):
        self.params = params
        self.dataset_name = dataset_name
        self.columns = columns
        self.name = None

    def load_data(self):
        # Load dataset from FastAPI endpoint
        response = requests.get(f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/" + self.dataset_name)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
        return df.to_numpy()

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

    @abstractmethod
    def run(self, data):
        # Abstract method to run the clustering algorithm
        pass

    def save_results(self, result, job_id, created_timestamp, started_timestamp, user_id, original_data=None):
        # Pop labels from result dictionary
        labels = result.pop("labels")

        # Optionally attach original data and column names
        # if original_data is not None:
        #     result["X"] = original_data.tolist()
        #     result["columns"] = self.columns

        print(f"job_id: {job_id}")
        print(f"dataset_name: {self.dataset_name}")
        print(f"columns: {self.columns}")
        print(f"created_timestamp: {created_timestamp}")
        print(f"started_timestamp: {started_timestamp}")
        print(f"finished_timestamp: {datetime.now(TIMEZONE).isoformat()}")
        print(f"clustering_algorithm: {self.name}")
        print(f"params: {self.params}")
        print(f"labels: {labels}")
        print(f"additional_results: {result}")
        print(f"user_id: {user_id}")

        # Post the result to FastAPI backend
        response = requests.post(
            f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/",
            json={
                "job_id": job_id,
                "dataset_name": self.dataset_name,
                "columns": self.columns,
                "created_timestamp": created_timestamp,
                "started_timestamp": started_timestamp,
                "finished_timestamp": datetime.now(TIMEZONE).isoformat(),
                "clustering_algorithm": self.name,
                "params": self.params,
                "labels": labels,
                "additional_results": result,
                "user_id": user_id
            }
        )
        return response
