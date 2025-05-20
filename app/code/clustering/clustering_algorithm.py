from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import requests
import os
import pandas as pd
import io
from datetime import datetime
import pytz

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

fastapi_host = os.getenv('FASTPI_HOST')
fastapi_port = os.getenv('FASTAPI_PORT')
fastapi_protocol = os.getenv('FASTAPI_PROTOCOL')

class ClusteringAlgorithm(ABC):
    def __init__(self, dataset_name, columns, **params):
        self.params = params
        self.dataset_name = dataset_name
        self.columns = columns
        self.name = None

    def load_data(self):
        response = requests.get(f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/dataset/" + self.dataset_name)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
        return df.to_numpy()

    def prepare_data(self, data, preprocess):
        X = data
        if preprocess:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        return X
    
    @abstractmethod
    def run(self, data):
        """Run the clustering algorithm and return result metadata."""
        pass

    def save_results(self, result, job_id, created_timestamp, started_timestamp, user_id):
        labels = result.pop("labels")
        return requests.post(
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


