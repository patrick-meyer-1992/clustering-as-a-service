import requests
import os
from time import sleep

FASTAPI_URL = "http://localhost:7002"

job_template = {
    "clustering_algorithm": "kmeans",
    "clustering_params": None,
    "columns": [
        {
        "name": "sepal.length",
        "type": "numeric"
        },
        {
        "name": "sepal.width",
        "type": "numeric"
        }
    ],
    "dataset_name": "iris.csv",
    "preprocess": "true",
    "preprocessing_params": {
        "feature_selection": "low_variance",
        "imputation_strategy": "mean",
        "normalization_type": "l2",
        "outlier_removal": "zscore",
        "outlier_threshold": 3,
        "pca_components": 10,
        "scaler": "auto",
        "transform_type": "quantile",
        "use_normalization": False,
        "use_pca": False,
        "variance_threshold": 0
    }
}

def test_root_endpoint():
    response = requests.get(f"{FASTAPI_URL}/datasets/")
    assert response.status_code == 200

def test_upload_iris():
    print(os.getcwd())
    files = {'file': open('./integration_tests/res/iris.csv', 'rb')}
    response = requests.put(f"{FASTAPI_URL}/dataset/", files=files)
    assert response.status_code == 200

def test_upload_numeric_and_text():
    files = {'file': open('./integration_tests/res/numeric_and_text.csv', 'rb')}
    response = requests.put(f"{FASTAPI_URL}/dataset/", files=files)
    assert response.status_code == 200

def test_duplicate_dataset():
    files = {'file': open('./integration_tests/res/iris.csv', 'rb')}
    response = requests.put(f"{FASTAPI_URL}/dataset/", files=files)
    assert response.status_code == 409  # Expecting a 409 Conflict for duplicate dataset

def test_get_iris():
    response = requests.get(f"{FASTAPI_URL}/dataset/iris.csv")
    assert response.status_code == 200

def test_get_numeric_and_text():
    response = requests.get(f"{FASTAPI_URL}/dataset/numeric_and_text.csv")
    assert response.status_code == 200

def test_get_non_existent_dataset():
    response = requests.get(f"{FASTAPI_URL}/dataset/non_existent.csv")
    assert response.status_code == 404  # Expecting a 404 Not Found for non-existent dataset

def test_list_datasets():
    response = requests.get(f"{FASTAPI_URL}/datasets/")
    assert response.status_code == 200
    datasets = response.json()
    assert isinstance(datasets, list)
    assert len(datasets) == 2  # Ensure there are datasets listed

def test_get_metadata_iris():
    for field in ["dataset_name", "columns", "size"]:
        response = requests.get(f"{FASTAPI_URL}/metadata/iris.csv?fields={field}")
        assert response.status_code == 200
        metadata = response.json()
        assert field in metadata

def test_get_metadata_all_fields():
    response = requests.get(f"{FASTAPI_URL}/metadata/iris.csv?fields=dataset_name&fields=columns&fields=size")
    assert response.status_code == 200
    metadata = response.json()
    assert "dataset_name" in metadata
    assert "columns" in metadata
    assert "size" in metadata

def test_post_job():
    response = requests.post(f"{FASTAPI_URL}/job/", json=job_template)
    assert response.status_code == 200
    job_response = response.json()
    assert "job_id" in job_response
    assert job_response["job_id"] is not None

    sleep(2)  # Wait for the job to be processed

    response = requests.get(f"{FASTAPI_URL}/result/{job_response['job_id']}/raw")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) > 0  # Ensure there are job statuses returned

def test_delete_iris():
    response = requests.delete(f"{FASTAPI_URL}/dataset/iris.csv")
    assert response.status_code == 200

def test_get_deleted_result():
    response = requests.get(f"{FASTAPI_URL}/result/12345/raw")
    assert response.status_code == 404  # Expecting a 404 Not Found for deleted result