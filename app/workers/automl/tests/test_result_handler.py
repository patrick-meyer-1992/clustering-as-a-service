import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock

from workers.automl.result_handler import send_results_to_backend


@patch("workers.automl.result_handler.requests.post")
@patch("workers.automl.result_handler.check_is_fitted")
def test_send_results_success(mock_check_is_fitted, mock_post):
    # Dummy DataFrame
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    # Dummy cluster with mocked predict()
    cluster = MagicMock()
    cluster.predict.return_value = np.array([0, 1])

    # Dummy model with get_params()
    model = MagicMock()
    model.get_params.return_value = {"n_clusters": 2}

    # Mock result dict
    result_dict = {
        "clustering_model": model,
        "optimal_cfg": {"algo": "KMeans"},
        "metafeatures_used": ["feat1", "feat2"],
        "metafeatures": np.array([0.1, 0.2]),
    }

    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    # Call function
    send_results_to_backend(
        job_id="abc123",
        dataset_name="test_dataset",
        columns=[{"name": "x", "type": "numeric"}, {"name": "y", "type": "numeric"}],
        created_timestamp="2025-01-01T12:00:00+00:00",
        started_timestamp="2025-01-01T12:01:00+00:00",
        result_dict=result_dict,
        cluster=cluster,
        df=df,
    )

    # Validate POST was called
    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/result/")
    assert kwargs["json"]["job_id"] == "abc123"
    assert kwargs["json"]["labels"] == [0, 1]
    assert kwargs["json"]["clustering_params"] == {"n_clusters": 2}


@patch("workers.automl.result_handler.requests.post")
@patch("workers.automl.result_handler.check_is_fitted")
def test_send_results_http_error(mock_check_is_fitted, mock_post):
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"

    cluster_mock = MagicMock()
    cluster_mock.predict.return_value = np.array([0, 1, 0])

    result_dict = {"clustering_model": MagicMock(), "optimal_cfg": None, "metafeatures_used": [], "metafeatures": None}
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(Exception, match="Error saving results"):
        send_results_to_backend(
            job_id="failjob456",
            dataset_name="test.csv",
            columns=[{"name": "x", "type": "numeric"}],
            created_timestamp="2024-01-01T00:00:00",
            started_timestamp="2024-01-01T00:01:00",
            result_dict=result_dict,
            cluster=cluster_mock,
            df=df,
        )


@patch("workers.automl.result_handler.requests.post")
@patch("workers.automl.result_handler.check_is_fitted", side_effect=ValueError("not fitted"))
def test_send_results_unfitted_model(mock_check_is_fitted, mock_post):
    mock_post.return_value.status_code = 200

    cluster_mock = MagicMock()
    cluster_mock.predict.return_value = [1, 0, 1]

    result_dict = {"clustering_model": MagicMock(), "optimal_cfg": {}, "metafeatures_used": [], "metafeatures": [1, 2]}
    df = pd.DataFrame({"x": [4, 5, 6]})

    send_results_to_backend(
        job_id="unfitted_model",
        dataset_name="dataset.csv",
        columns=[{"name": "x", "type": "numeric"}],
        created_timestamp="2024-01-01T00:00:00",
        started_timestamp="2024-01-01T00:01:00",
        result_dict=result_dict,
        cluster=cluster_mock,
        df=df,
    )

    payload = mock_post.call_args[1]["json"]
    assert payload["clustering_params"] == {}


