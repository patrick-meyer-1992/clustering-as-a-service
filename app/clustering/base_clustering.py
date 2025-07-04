import io
import os
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from typing import Literal
from pydantic import BaseModel, Field

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

# Environment variables for FastAPI connection
FASTAPI_HOST = os.getenv("FASTAPI_HOST")
FASTAPI_PORT = os.getenv("FASTAPI_PORT")
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL")

class PreProcessingParams(BaseModel):
    scaler: Literal["auto"] = Field(
        title="The type of scaler to use for preprocessing",
        default="auto",
        description="Choose 'auto' to let the algorithm decide the best scaler based on the dataset."
    )
    use_normalization: bool = Field(
        title="Whether to apply normalization to the data",
        default=False,
        description="Set to True to apply normalization."
    )
    normalization_type: Literal["l1", "l2", "max"] = Field(
        title="The type of normalization to apply",
        default="l2",
        description="Choose 'l1', 'l2' or 'max' for the normalization type."
    )
    use_pca: bool = Field(
        title="Whether to apply PCA for dimensionality reduction",
        default=False,
        description="Set to True to apply PCA."
    )
    pca_components: int = Field(
        title="Number of components to keep after PCA",
        default=10,
        description="Choose the number of components to keep after PCA."
    )
    transform_type: Literal[None] = Field(
        title="The type of transformation to apply to the data",
        default=None,
        description="Choose 'log', 'sqrt', 'power' or None for no transformation."
    )
    imputation_strategy: Literal["mean", "median", "none"] = Field(
        title="Strategy for handling missing values",
        default="mean",
        description="Choose 'mean', 'median' or 'none' for the imputation strategy."
    )
    outlier_removal: Literal["none", "zscore", "iqr"] = Field(
        title="Method for outlier removal",
        default="none",
        description="Choose 'none', 'zscore' or 'iqr' for the outlier removal method."
    )
    outlier_threshold: float = Field(
        title="Threshold for outlier detection",
        default=3.0,
        description="Choose the threshold for outlier detection. Only used if outlier_removal is 'zscore'."
    )
    feature_selection: Literal["none", "low_variance", "constant"] = Field(
        title="Method for feature selection", 
        default="none",
        description="Choose 'none', 'low_variance' or 'constant' for the feature selection method."
    )
    variance_threshold: float = Field(
        title="Variance threshold for feature selection",
        default=0.0,
        description="Choose the variance threshold for feature selection. Only used if feature_selection is 'low_variance'."
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                "scaler": "auto",
                "use_normalization": False,
                "normalization_type": "l2",
                "use_pca": False,
                "pca_components": 10,
                "transform_type": None,
                "imputation_strategy": "none",
                "outlier_removal": "none",  # "none", "zscore", "iqr"
                "outlier_threshold": 3.0,  # only for zscore
                "feature_selection": "none",  # "none", "low_variance", "constant"
                "variance_threshold": 0.0,  # for low_variance
                }
            ]
        }
    }

DEFAULT_PREPROCESSING_PARAMS = {
    "scaler": "auto",
    "use_normalization": False,
    "normalization_type": "l2",
    "use_pca": False,
    "pca_components": 10,
    "transform_type": None,
    "imputation_strategy": "none",
    "outlier_removal": "none",  # "none", "zscore", "iqr"
    "outlier_threshold": 3.0,  # only for zscore
    "feature_selection": "none",  # "none", "low_variance", "constant"
    "variance_threshold": 0.0,  # for low_variance
}


class BaseClustering(ABC):
    frontend_name = None
    backend_name = None

    def __init__(self, dataset_name, columns, preprocessing_params = None, **clustering_params):
        self.clustering_params = clustering_params
        self.dataset_name = dataset_name
        self.columns = columns
        self.preprocessing_params = preprocessing_params
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

    def validate_params(self):
        """
        Basic validation of input parameters.
        This does not replace sklearn's internal validation,
        but catches obvious issues early (e.g. wrong types, empty values).
        """
        for key, value in self.clustering_params.items():
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

        # Set default preprocessing parameters if not provided
        params = self.preprocessing_params if self.preprocessing_params is not None else DEFAULT_PREPROCESSING_PARAMS

        # Determine scaler
        scaler_type = params["scaler"]
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
        if params["use_normalization"]:
            normalizer = Normalizer(norm=params["normalization_type"])
            X_scaled = normalizer.fit_transform(X_scaled)

        # Optional imputation (missing value handling)
        imputation_strategy = params["imputation_strategy"]

        if imputation_strategy == "mean":
            df = pd.DataFrame(X_scaled)
            X_scaled = df.fillna(df.mean()).to_numpy()
        elif imputation_strategy == "median":
            df = pd.DataFrame(X_scaled)
            X_scaled = df.fillna(df.median()).to_numpy()
        elif imputation_strategy == "none":
            pass
        else:
            raise ValueError(f"Unsupported imputation strategy: {imputation_strategy}")

        # Optional outlier removal
        outlier_strategy = params["outlier_removal"]
        threshold = params["outlier_threshold"]

        if outlier_strategy == "zscore":
            z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
            mask = (z_scores < threshold).all(axis=1)
            X_scaled = X_scaled[mask]
        elif outlier_strategy == "iqr":
            Q1 = np.percentile(X_scaled, 25, axis=0)
            Q3 = np.percentile(X_scaled, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = ((X_scaled >= lower) & (X_scaled <= upper)).all(axis=1)
            X_scaled = X_scaled[mask]
        elif outlier_strategy == "none":
            pass
        else:
            raise ValueError(f"Unsupported outlier removal strategy: {outlier_strategy}")

        # Optional feature selection
        feature_sel = params["feature_selection"]
        var_thresh = params["variance_threshold"]

        if feature_sel == "constant":
            selector = VarianceThreshold(threshold=0.0)
            X_scaled = selector.fit_transform(X_scaled)
        elif feature_sel == "low_variance":
            selector = VarianceThreshold(threshold=var_thresh)
            X_scaled = selector.fit_transform(X_scaled)
        elif feature_sel == "none":
            pass
        else:
            raise ValueError(f"Unsupported feature_selection method: {feature_sel}")

        # Optional PCA dimensionality reduction
        if params["use_pca"] and X_scaled.shape[1] > params["pca_components"]:
            pca = PCA(n_components=params["pca_components"])
            X_scaled = pca.fit_transform(X_scaled)

        # Optional post-scaling transformation
        transform_type = params["transform_type"]

        if transform_type == "quantile":
            transformer = QuantileTransformer(output_distribution="normal")
            X_scaled = transformer.fit_transform(X_scaled)
        elif transform_type == "power":
            transformer = PowerTransformer()
            X_scaled = transformer.fit_transform(X_scaled)
        elif transform_type not in (None, ""):
            raise ValueError(f"Unsupported transform_type: {transform_type}")

        return X_scaled

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
                "clustering_algorithm": self.frontend_name,
                "clustering_params": self.clustering_params,
                "preprocessing_params": self.preprocessing_params,
                "labels": labels,
                "additional_results": result,
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
