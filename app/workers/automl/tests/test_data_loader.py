import pandas as pd
import pytest
from unittest.mock import patch, Mock
from workers.automl.data_loader import fetch_dataset


def test_fetch_dataset_success():
    # Setup
    job_id = "123"
    dataset_name = "mock_dataset"
    columns = [{"name": "col1"}, {"name": "col2"}]
    csv_data = b"col1,col2,col3\n1,a,x\n2,b,y"

    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = csv_data
        mock_get.return_value = mock_response

        df = fetch_dataset(job_id, dataset_name, columns)

        assert list(df.columns) == ["col1", "col2"]
        assert df.shape == (2, 2)
        assert df["col1"].tolist() == [1, 2]
        assert df["col2"].tolist() == ["a", "b"]


def test_fetch_dataset_http_error():
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            fetch_dataset("jobX", "broken_dataset", [{"name": "x"}])


def test_fetch_dataset_missing_column():
    csv_data = b"a,b\n1,2\n3,4"
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = csv_data
        mock_get.return_value = mock_response

        with pytest.raises(KeyError):
            fetch_dataset("jobY", "some_dataset", [{"name": "non_existing"}])

