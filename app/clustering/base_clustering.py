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
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
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
    """
    Abstract base class for all clustering algorithm implementations.

    This class defines the interface and shared functionality for clustering wrappers,
    including data loading, encoding, preprocessing, result saving, and quality metric computation.

    Subclasses must implement the `run` and `get_default_params` methods.
    """

    frontend_name = None
    backend_name = None

    def __init__(self, dataset_name, columns, preprocessing_params=None, **clustering_params):
        """
        Initialize the clustering wrapper.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load via FastAPI.
        columns : list[dict[str, str]]
            List of dictionaries specifying the name and type of each feature column.
        preprocessing_params : dict, optional
            Dictionary of preprocessing options (scaling, PCA, etc.).
        **clustering_params : dict
            Algorithm-specific clustering parameters.
        """

        self.clustering_params = clustering_params
        self.dataset_name = dataset_name
        self.columns: list[dict[str, str]] = columns
        self.preprocessing_params = preprocessing_params
        self.name = "Base Clustering"  # Default name

        # Set default preprocessing parameters if not provided
        for k, v in self.get_default_params().items():
            if k not in self.clustering_params:
                self.clustering_params[k] = v

    def load_data(self):
        """
        Fetch the dataset from the FastAPI backend using the provided dataset name.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the selected columns.

        Raises
        ------
        Exception
            If there is a problem with the network request or data loading.
        """

        try:
            url = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}/dataset/{self.dataset_name}"
            print(f"Trying to load data from: {url}")  # Debug
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
            # Use only selected columns
            return df[[col.get("name") for col in self.columns]]
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns in the dataset using appropriate encoders.

        - Ordinal columns are encoded with `OrdinalEncoder` using specified or inferred order.
        - Nominal columns are encoded with `OneHotEncoder`.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with raw categorical features.

        Returns
        -------
        pd.DataFrame
            The DataFrame with encoded categorical columns.
        """

        ordinal_cols = [col.get("name") for col in self.columns if col.get("type") == "ordinal"]
        nominal_cols = [col.get("name") for col in self.columns if col.get("type") == "nominal"]

        if ordinal_cols:
            # Prepare user-defined categories
            categories = []
            for col in ordinal_cols:
                col_info = next((c for c in self.columns if c["name"] == col), None)
                if col_info and "order" in col_info:
                    categories.append(col_info["order"])
                else:
                    # If no order is provided, fallback to default inferred ordering
                    unique_values = sorted(df[col].dropna().unique().tolist())
                    categories.append(unique_values)

            encoder = OrdinalEncoder(categories=categories)
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
        """
        Optionally apply preprocessing pipeline to the given dataset.

        If `preprocess=True`, the method initializes the `PreprocessingPipeline` using the
        provided or default parameters, then fits and transforms the data.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            The input data to preprocess.
        preprocess : bool, default=True
            Whether to apply the preprocessing pipeline.

        Returns
        -------
        np.ndarray
            The preprocessed data.

        Raises
        ------
        ValueError
            If the preprocessing parameters are invalid.
        """

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
        Return a dictionary of default parameters for the clustering algorithm.

        This method must be implemented by each subclass.

        Returns
        -------
        dict
            A dictionary of default hyperparameters specific to the algorithm.
        """

        pass

    @abstractmethod
    def run(self, data):
        """
        Execute the clustering algorithm on the given data.

        This method must be implemented by each subclass to apply the specific clustering logic.

        Parameters
        ----------
        data : np.ndarray
            The input data on which clustering will be performed.

        Returns
        -------
        dict
            A dictionary containing at least the cluster labels and any additional results or metrics.
        """

        pass

    def compute_quality_metrics(self, data, labels):
        """
        Compute standard clustering quality metrics, ignoring noise labels (e.g. -1).

        Metrics calculated:
        - Silhouette Score
        - Davies-Bouldin Score
        - Calinski-Harabasz Score

        If all labels are the same or there's only one cluster, returns None for all metrics.

        Parameters
        ----------
        data : np.ndarray
            The input data used for clustering.
        labels : np.ndarray
            Cluster labels assigned to each data point.

        Returns
        -------
        dict
            Dictionary containing values for 'silhouette_score', 'davies_bouldin_score',
            and 'calinski_harabasz_score'. Metrics are None if not computable.
        """

        try:
            filtered_data = data[labels >= 0]
            filtered_labels = labels[labels >= 0]

            if len(set(filtered_labels)) <= 1:
                return {"silhouette_score": None, "davies_bouldin_score": None, "calinski_harabasz_score": None}

            return {
                "silhouette_score": silhouette_score(filtered_data, filtered_labels),
                "davies_bouldin_score": davies_bouldin_score(filtered_data, filtered_labels),
                "calinski_harabasz_score": calinski_harabasz_score(filtered_data, filtered_labels),
            }
        except Exception:
            return {"silhouette_score": None, "davies_bouldin_score": None, "calinski_harabasz_score": None}

    def save_results(self, result, job_id, created_timestamp, started_timestamp):
        """
        Send clustering results and metadata to the FastAPI backend.

        This method extracts the labels, packages the clustering results along with job
        and preprocessing metadata, and sends a POST request to the backend.

        Parameters
        ----------
        result : dict
            Dictionary containing the clustering output. Must include a "labels" key.
        job_id : str
            The ID of the job for tracking purposes.
        created_timestamp : str
            ISO-formatted creation timestamp of the job.
        started_timestamp : str
            ISO-formatted timestamp when clustering began.

        Returns
        -------
        dict or None
            JSON response from the FastAPI backend if successful, otherwise None.
        """

        try:
            print(f"Saving results for job_id: {job_id}")  # Debug print

            # Pop labels from result dictionary
            labels = result.pop("labels")

            payload = {
                "job_id": job_id,
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
        """
        Recursively sanitize data structures by converting NaN and infinite values to strings.

        This ensures compatibility with JSON serialization and avoids FastAPI errors
        when posting payloads containing non-finite values.

        Parameters
        ----------
        obj : any
            Any nested structure (dict, list, numpy types) potentially containing inf/nan.

        Returns
        -------
        any
            A sanitized copy of the input with inf/nan replaced as strings ("inf", "-inf", "nan").
        """

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
