import numpy as np
import uuid
import os
import joblib
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import logging

class ClusteringAlgorithm(ABC):
    def __init__(self, data, preprocess=True):
        self.raw_data = np.array(data)
        self.preprocess = preprocess
        self.job_id = str(uuid.uuid4())
        self.model = None
        self.labels = None

    def _prepare_data(self):
        print(f"Preparing data for job {self.job_id}")
        X = self.raw_data
        if self.preprocess:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        return X

    def _save_results(self):
        pass
        # os.makedirs("results", exist_ok=True)
        # model_path = f"results/model_{self.job_id}.pkl"
        # labels_path = f"results/labels_{self.job_id}.npy"
        # joblib.dump(self.model, model_path)
        # np.save(labels_path, self.labels)
        # return model_path, labels_path

    @abstractmethod
    def run(self):
        """Run the clustering algorithm and return result metadata."""
        pass
