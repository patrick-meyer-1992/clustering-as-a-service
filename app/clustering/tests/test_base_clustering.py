import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Make sure the app directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.clustering.base_clustering import BaseClustering


class DummyClustering(BaseClustering):
    backend_name = "Dummy"
    frontend_name = "Dummy"

    @staticmethod
    def get_default_params():
        return {}

    def run(self, data):
        return {"labels": np.array([0, 1, 1, 0]), "some_metric": 0.95}


@pytest.fixture
def dummy_instance():
    columns = [
        {"name": "feature1", "type": "numeric"},
        {
            "name": "feature2",
            "type": "ordinal",
            "order": ["low", "medium", "high"],
        },
    ]

    return DummyClustering(
        dataset_name="test_dataset.csv",
        columns=columns,
        preprocessing_params={},
    )


def test_encode_data_with_ordinal(dummy_instance):
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["low", "medium", "high"]})
    encoded = dummy_instance.encode_data(df.copy())
    assert "feature2" in encoded.columns
    assert encoded["feature2"].between(0, 2).all()


def test_encode_data_with_nominal():
    from clustering.tests.test_base_clustering import DummyClustering

    dummy = DummyClustering(
        dataset_name="test.csv",
        columns=[{"name": "color", "type": "nominal"}],
        preprocessing_params={},
        clustering_algorithm="dummy",
    )

    df = pd.DataFrame({"color": ["red", "green", "blue", "green"]})

    encoded = dummy.encode_data(df)

    assert "color" not in encoded.columns
    assert any(col.startswith("color_") for col in encoded.columns)
    assert encoded.shape[0] == 4


def test_encode_data_mixed(dummy_instance):
    df = pd.DataFrame(
        {
            "feature1": ["low", "medium", "high"],  # ordinal
            "feature2": ["red", "blue", "green"],  # nominal
        }
    )

    dummy_instance.columns = [
        {"name": "feature1", "type": "ordinal", "order": ["low", "medium", "high"]},
        {"name": "feature2", "type": "nominal"},
    ]

    encoded = dummy_instance.encode_data(df.copy())

    assert "feature1" in encoded.columns
    assert encoded["feature1"].isin([0, 1, 2]).all()
    assert all(col.startswith("feature2_") for col in encoded.columns if col.startswith("feature2_"))
    assert "feature2" not in encoded.columns


def test_compute_quality_metrics_valid(dummy_instance):
    data = np.array([[0, 0], [1, 1], [0, 1]])
    labels = np.array([0, 1, 0])
    metrics = dummy_instance.compute_quality_metrics(data, labels)

    assert all(k in metrics for k in ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"])
    assert all(isinstance(metrics[k], float) for k in metrics if metrics[k] is not None)


def test_compute_quality_metrics_invalid(dummy_instance):
    data = np.array([[0], [1], [2]])
    labels = np.array([0, 0, 0])
    metrics = dummy_instance.compute_quality_metrics(data, labels)
    assert metrics["silhouette_score"] is None


def test_sanitize_inf_and_nan(dummy_instance):
    test_obj = {
        "a": float("inf"),
        "b": float("-inf"),
        "c": float("nan"),
        "d": np.array([1, 2, float("inf")]),
    }
    sanitized = dummy_instance._sanitize_inf(test_obj)
    assert sanitized["a"] == "inf"
    assert sanitized["b"] == "-inf"
    assert sanitized["c"] == "nan"
    assert sanitized["d"][-1] == "inf"


@patch("app.clustering.base_clustering.requests.post")
def test_save_results_success(mock_post, dummy_instance):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": "ok"}

    result = {"labels": [0, 1, 1, 0], "some_metric": 0.9}
    resp = dummy_instance.save_results(
        result, job_id="abc123", created_timestamp="2025-01-01T00:00:00Z", started_timestamp="2025-01-01T00:01:00Z"
    )
    assert resp["status"] == "ok"


@patch("app.clustering.base_clustering.requests.post")
def test_save_results_failure(mock_post, dummy_instance):
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"

    result = {"labels": [0, 1], "some_metric": 0.1}
    with pytest.raises(RuntimeError, match="Internal Server Error"):
        dummy_instance.save_results(
            result,
            job_id="job_fail",
            created_timestamp="2025-01-01T00:00:00Z",
            started_timestamp="2025-01-01T00:01:00Z",
        )


@patch("app.clustering.base_clustering.requests.post", side_effect=Exception("Connection failed"))
def test_save_results_exception(mock_post, dummy_instance):
    result = {"labels": [0, 1], "some_metric": 0.1}
    with pytest.raises(RuntimeError):
        dummy_instance.save_results(
            result,
            job_id="job_fail",
            created_timestamp="2025-01-01T00:00:00Z",
            started_timestamp="2025-01-01T00:01:00Z",
        )


@patch("app.clustering.base_clustering.requests.get")
def test_load_data_success(mock_get, dummy_instance):
    mock_csv = b"feature1,feature2\n1,low\n2,medium\n3,high"
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = mock_csv

    df = dummy_instance.load_data()
    assert list(df.columns) == ["feature1", "feature2"]
    assert df.shape[0] == 3


def test_prepare_data_no_preprocessing(dummy_instance):
    data = np.random.rand(5, 2)
    out = dummy_instance.prepare_data(data, preprocess=False)
    assert np.allclose(out, data)
