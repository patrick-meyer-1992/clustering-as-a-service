from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import requests
import os
import pandas as pd
import io
from datetime import datetime
import pytz
import json

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

fastapi_host = os.getenv('FASTPI_HOST')
fastapi_port = os.getenv('FASTAPI_PORT')
fastapi_protocol = os.getenv('FASTAPI_PROTOCOL')

class BaseClustering(ABC):
    def __init__(self, dataset_name, columns, **params):
        self.params = params
        self.dataset_name = dataset_name
        self.columns = columns if isinstance(columns, list) else json.loads(columns)
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

    def save_results(self, result, job_id, created_timestamp, started_timestamp, user_id, original_data=None):
        """
        Save clustering results to FastAPI backend
        """
        try:
            print(f"Saving results for job_id: {job_id}")  # Debug print
            labels = result.pop("labels")

            payload = {
                "job_id": job_id,  # Hier verwenden wir den übergebenen job_id Parameter
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

            # Add original data if provided
            if original_data is not None:
                payload["additional_results"]["X"] = original_data.tolist()
                payload["additional_results"]["columns"] = self.columns

            # Send results to FastAPI backend
            url = f"{fastapi_protocol}://{fastapi_host}:{fastapi_port}/result/"
            print(f"Sending results to: {url}")  # Debug print
            response = requests.post(url, json=payload)
            
            if response.status_code != 200:
                print(f"Error saving results: {response.text}")
                return None
                
            return response.json()

        except Exception as e:
            print(f"Error in save_results: {str(e)}")
            return None



