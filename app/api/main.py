import io
import json
import os
from datetime import datetime
from typing import Any, Literal, Annotated

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
from fastapi import Depends, FastAPI, File, Path, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pymongo import AsyncMongoClient
from workers.celery_conn import celery
from workers.tasks import run_clustering_job
from clustering import wrappers
from clustering.base_clustering import PreProcessingParams

app = FastAPI()

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

ALGORITHM_MAP = {
    getattr(wrappers, algo).backend_name: getattr(wrappers, algo)
    for algo in dir(wrappers)
    if algo.endswith("Wrapper")
}
algorithms = [backend_name for backend_name in ALGORITHM_MAP.keys()]
algorithms.append("auto")

def validate_data(data):
    # Check for None
    if data is None:
        raise ValueError("Input data is None.")

    # Check shape attribute
    if not hasattr(data, "shape"):
        raise TypeError("Input data must be array-like with a shape attribute.")

    # Check empty
    if data.size == 0:
        raise ValueError("Input data is empty.")

    # Check 2D shape
    if len(data.shape) != 2:
        raise ValueError("Input data must be 2-dimensional (samples, features).")

    # Check if numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Input data must be numeric.")

async def get_mongodb():
    MONGODB_DB = os.getenv("MONGODB_DB")
    MONGODB_HOST = os.getenv("MONGODB_HOST")
    MONGODB_PORT = os.getenv("MONGODB_PORT")

    # For testing purposes, you can set MONGODB_URL in your environment variables
    MONGODB_URL = os.getenv("MONGODB_URL", None)

    if MONGODB_URL:
        mongodb_client = AsyncMongoClient(MONGODB_URL)
        print(f"Using MongoDB URL: {MONGODB_URL}")
    else:
        mongodb_client = AsyncMongoClient(f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}")

    mongodb_database = mongodb_client.get_database(MONGODB_DB)
    try:
        yield mongodb_database
    finally:
        await mongodb_client.close()

@app.get("/dataset/{dataset_name}", response_class=StreamingResponse)
async def get_dataset(dataset_name: str, mongodb_database=Depends(get_mongodb)):
    try:
        # Debugging: Logge den übergebenen dataset_name
        print(f"Retrieving dataset: {dataset_name}")

        # Hole den Datensatz aus MongoDB
        data_collection = mongodb_database.get_collection("data")
        dataset = await data_collection.find_one({"dataset_name": dataset_name}, {"_id": 0, "data": 1})

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Debugging: Logge die abgerufenen Daten
        print(f"Dataset retrieved: {dataset}")

        # Erstelle einen StreamingResponse für die CSV-Daten
        file_stream = io.StringIO(dataset["data"])
        return StreamingResponse(
            file_stream,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}"},
        )
    except Exception as e:
        print(f"Error retrieving dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

class DatasetPutResponse(BaseModel):
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[str] = Field(description="The columns used in the dataset", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"]
                }
            ]
        }
    }
@app.put("/dataset/", response_model=DatasetPutResponse)
async def put_dataset(
    file: UploadFile = File(description="The CSV file to upload", default=...),
    mongodb_database=Depends(get_mongodb),
):
    data_collection = mongodb_database.get_collection("data")
    # Check if file with the same name already exists
    exists = await data_collection.find_one({"dataset_name": file.filename})
    if exists:
        raise HTTPException(status_code=409, detail="Dataset with this name already exists.")
    try:
        content = await file.read()

        # Parse the CSV file to extract column names
        try:
            df = pd.read_csv(io.BytesIO(content))
            columns = df.columns.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}") from e

        # Convert to Numpy array and validate
        data_array = df.to_numpy()
        validate_data(data_array)

        # Debugging: Logge die zu speichernden Daten
        print(f"Saving dataset: {file.filename}, columns: {columns}")

        # Store the dataset and metadata in MongoDB
        await data_collection.insert_one(
            {
                "dataset_name": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "columns": columns,
                "data": content.decode("utf-8"),  # Store the CSV content as a string
            }
        )

        return {"dataset_name": file.filename, "columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

class DatasetDeleteResponse(BaseModel):
    dataset_name: str = Field(description="The name of the deleted dataset", default=...)
    job_ids: list[str] | None = Field(description="List of job IDs associated with the deleted dataset", default=None)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "job_ids": ["fb231936-1d83-43de-85a4-81c6889dd21c", "777b6e2e-07b6-47a8-82e1-f08900ea0176"]
                }
            ]
        }
    }

@app.delete("/dataset/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    mongodb_database=Depends(get_mongodb),
) -> DatasetDeleteResponse:
    """
    Delete a dataset from mongoDB and all results associated with it.
    """
    data_collection = mongodb_database.get_collection("data")
    result_collection = mongodb_database.get_collection("results")
    # Get all job IDs associated with the dataset
    jobs = await result_collection.find({"dataset_name": dataset_name}, {"_id": 0,"job_id": 1}).to_list(length=None)
    await result_collection.delete_many({"dataset_name": dataset_name})
    result = await data_collection.delete_one({"dataset_name": dataset_name})

    if result.deleted_count == 1:
        return DatasetDeleteResponse(dataset_name=dataset_name, job_ids=[job["job_id"] for job in jobs])
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")

class DatasetGetResponse(BaseModel):
    dataset_name: str = Field(description="The name of the uploaded dataset", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv"
                }
            ]
        }
    }

@app.get("/datasets/")
async def get_datasets(mongodb_database=Depends(get_mongodb)) -> list[DatasetGetResponse]:
    """
    Returns a list of all uploaded datasets.
    """
    data_collection = mongodb_database.get_collection("data")
    datasets = await data_collection.find({}, {"_id": 0, "dataset_name": 1}).to_list(length=1000)
    return [DatasetGetResponse(**dataset) for dataset in datasets]

class JobPostRequest(BaseModel):
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[str] | None = Field(description="The columns to use for clustering", default=None)
    clustering_algorithm: str = Field(description="The clustering algorithm to use", default=..., examples=algorithms)
    preprocess: bool = Field(description="Whether to preprocess the data", default=True)
    clustering_params: dict[str, Any] | None = Field(description="Clustering algorithm parameters", default=None)
    preprocessing_params: PreProcessingParams | None = Field(description="Preprocessing parameters", default=None)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"],
                    "clustering_algorithm": "kmeans",
                    "preprocess": "true",
                    "clustering_params": {"n_clusters": 3},
                    "preprocessing_params": {
                        "scaler": "auto",
                        "use_normalization": False,
                        "normalization_type": "l2",
                        "use_pca": False,
                        "pca_components": 10,
                        "transform_type": None,
                        "imputation_strategy": "none",
                        "outlier_removal": "none", 
                        "outlier_threshold": 3.0,
                        "feature_selection": "none",
                        "variance_threshold": 0.0, 
                    }
                }
            ]
        }
    }

class JobPostResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c"
                }
            ]
        }
    }

@app.post("/job/")
async def post_job(req: JobPostRequest) -> JobPostResponse:
    try:
        job = run_clustering_job.delay(
            req.dataset_name,
            req.columns,
            datetime.now(TIMEZONE).isoformat(),
            req.clustering_algorithm,
            req.preprocess,
            preprocessing_params = req.preprocessing_params.model_dump() if req.preprocessing_params else None,
            **(req.clustering_params or {}),
        )

        return JobPostResponse(
            job_id=job.id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting clustering: {str(e)}") from e

class JobsGetResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    dataset_name: str = Field(description="The name of the dataset", default=...)
    created_timestamp: str = Field(description="The creation timestamp", default=...)
    clustering_algorithm: str = Field(description="The clustering algorithm used", default=...)
    status: str | None = Field(description="The status of the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                    "created_timestamp": "2025-07-02T11:53:56.083632+00:00",
                    "clustering_algorithm": "kmeans",
                    "status": "SUCCESS"
                }
            ]
        }
    }
@app.get("/jobs/")
async def get_jobs(mongodb_database=Depends(get_mongodb)) -> list[JobsGetResponse]:
    """
    Returns an overview of all known jobs including their status.
    """
    try:
        results_collection = mongodb_database.get_collection("results")
        jobs = await results_collection.find({}, {"_id": 0}).to_list(length=1000)
        job_list = []
        for job in jobs:
            job_id = job.get("job_id")
            celery_status = None
            if job_id:
                task = celery.AsyncResult(job_id)
                celery_status = task.status
            job_list.append(
                JobsGetResponse(
                    job_id=job_id,
                    dataset_name=job.get("dataset_name"),
                    created_timestamp=job.get("created_timestamp"),
                    clustering_algorithm=job.get("clustering_algorithm"),
                    status=celery_status
                )
            )
        return job_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {e}") from e

@app.get("/result/{job_id}/table")
async def get_result_table(
    job_id: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": job_id})
        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")

        additional = result.get("additional_results", {})
        labels = result.get("labels")
        X = additional.get("X")
        columns = additional.get("columns")

        if X is not None and columns is not None:
            df = []
            for row, label in zip(X, labels, strict=False):
                row_dict = {col: val for col, val in zip(columns, row, strict=False)}
                row_dict["Cluster"] = label
                df.append(row_dict)
            return {"data": df, "columns": columns + ["Cluster"]}
        else:
            return {"labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

@app.get("/result/{job_id}/raw")
async def get_result_raw(
    job_id: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": job_id})
        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")
        return result.get("labels")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

@app.get("/result/{job_id}/graph")
async def get_result_graph(
    job_id: str,
    x_column: str | None = Query(None, title="The column to use for the x-axis. Set only in combination with y_column"),
    y_column: str | None = Query(None, title="The column to use for the y-axis. Set only in combination with x_column"),
    mongodb_database=Depends(get_mongodb),
):
    
    """
    Creates a 2D plotly scatter plot for the clustering result of a job.

    If x_column or y_column are not provided, the first two columns of the result will be used.    
    """
    try:
        result_collection = mongodb_database.get_collection("results")
        data_collection = mongodb_database.get_collection("data")

        result = await result_collection.find_one({"job_id": job_id})
        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")
        
        dataset = await data_collection.find_one({"dataset_name": result.get("dataset_name")}, {"_id": 0, "data": 1})
        df = pd.read_csv(io.StringIO(dataset["data"]))
        labels = [str(label) for label in sorted(result.get("labels"))]
        if df is None or labels is None or df.shape[0] == 0 or df.shape[1] < 2 or len(labels) == 0:
            raise HTTPException(status_code=400, detail="No data for plotting")
        
        if x_column is None or y_column is None:
            x_column = result.get("columns")[0]
            y_column = result.get("columns")[1]
        elif x_column not in df.columns or y_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid x_column or y_column")
        
        df = df[[x_column, y_column]]

        print(df.head())

        if df.shape[1] != 2:
            raise HTTPException(status_code=400, detail="Data is not 2D")
        fig = px.scatter(
            x=df[x_column],
            y=df[y_column],
            color=labels,
            title=f"Clustering: {result.get('clustering_algorithm')}",
            labels={"x": x_column, "y": y_column}
        )
        return fig.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e
class ResultPostRequest(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[str] = Field(description="The columns used in the dataset", default=...)
    created_timestamp: str = Field(description="The creation timestamp", default=...)
    started_timestamp: str = Field(description="The start timestamp", default=...)
    finished_timestamp: str = Field(description="The finish timestamp", default=...)
    clustering_algorithm: str = Field(description="The clustering algorithm used", default=...)
    clustering_params: dict[str, Any] = Field(description="The parameters for the clustering algorithm", default=...)
    preprocessing_params: dict[str, Any] = Field(description="The parameters for the preprocessing", default=...)
    labels: list[int | None] = Field(description="The labels for the dataset", default=...)
    additional_results: dict[str, Any] = Field(description="Additional results from the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"],
                    "created_timestamp": "2025-07-02T11:53:56.083632+00:00",
                    "started_timestamp": "2025-07-02T11:54:56.083632+00:00",
                    "finished_timestamp": "2025-07-02T11:55:56.083632+00:00",
                    "clustering_algorithm": "kmeans",
                    "clustering_params": {"n_clusters": 3},
                    "preprocessing_params": {"scaler": "standard"},
                    "labels": [0, 1, 2, 0, 1, 2],
                    "additional_results": {
                        "centers": [
                        [
                            -0.11190209211560795,
                            -0.9579796406026493
                        ],
                        [
                            1.0961653346135656,
                            0.08900941628667573
                        ],
                        [
                            -1.0020665312812713,
                            0.9062549154367601
                        ]
                        ],
                        "n_iter": 16,
                    }
                }
            ]
        }
    }

class ResultPostResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c"
                }
            ]
        }
    }

@app.post("/result/")
async def post_result(req: ResultPostRequest, mongodb_database=Depends(get_mongodb)) -> ResultPostResponse:
    try:
        result_collection = mongodb_database.get_collection("results")
        # Debugging: Logge die zu speichernden Daten
        print(f"Saving result: {req.model_dump()}")
        await result_collection.insert_one(req.model_dump())
        return ResultPostResponse(job_id=req.job_id)
    except Exception as e:
        print(f"Error saving result: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving result: {e}") from e

class AutoMlClusterRequest(BaseModel):
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[str] = Field(description="The columns to use for clustering", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"]
                }
            ]
        }
    }

class AutoMlClusterResponse(BaseModel):
    job_id: str = Field(description="The ID of the AutoML clustering job", default=...)
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[str] = Field(description="The columns used for clustering", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"]
                }
            ]
        }
    }
@app.post("/automl/job")
async def start_automl(req: AutoMlClusterRequest) -> AutoMlClusterResponse:
    print("[AutoML] Received new request on /automl/job")

    try:
        print(f"[AutoML] Dataset: {req.dataset_name}")
        print(f"[AutoML] Columns: {req.columns}")

        job = celery.send_task(
            "automl_worker.run_autocluster", kwargs={"dataset_name": req.dataset_name, "columns": req.columns}
        )

        print(f"[AutoML] Job started with ID: {job.id}")

        return AutoMlClusterResponse(
            job_id=job.id,
            dataset_name=req.dataset_name,
            columns=req.columns,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting AutoML-Job: {str(e)}")

@app.get("/automl/result/")
async def get_automl_result(
    job_id: str = Query(...),
    presentation: str = Query("table", enum=["table", "raw", "graph"]),
    mongodb_database=Depends(get_mongodb),
):
    result_collection = mongodb_database.get_collection("results")
    result = await result_collection.find_one({"job_id": job_id, "clustering_algorithm": "AutoCluster"})

    if not result:
        raise HTTPException(status_code=404, detail=f"No result found for job_id: {job_id}")

    labels = result.get("labels")
    additional = result.get("additional_results", {})
    X = additional.get("X")
    columns = additional.get("columns")

    if presentation == "table":
        if X and columns:
            df = []
            for row, label in zip(X, labels, strict=False):
                row_dict = {col: val for col, val in zip(columns, row, strict=False)}
                row_dict["Cluster"] = label
                df.append(row_dict)
            return {"data": df, "columns": columns + ["Cluster"]}
        else:
            return {"labels": labels}

    if presentation == "raw":
        return labels

    if presentation == "graph":
        if not X or not labels or len(X[0]) < 2:
            raise HTTPException(status_code=400, detail="Graph data requires at least 2D input.")
        fig = px.scatter(
            x=[row[0] for row in X],
            y=[row[1] for row in X],
            color=[str(label) for label in labels],
            title=f"AutoML Clustering: {result.get('clustering_algorithm')}",
        )
        return fig.to_dict()

    raise HTTPException(status_code=400, detail="Invalid presentation format")

@app.get("/debug/job/{job_id}")
async def debug_job(job_id: str, mongodb_database=Depends(get_mongodb)):
    """
    Debug endpoint to check job status and results
    """
    try:
        # Check Celery task
        task = celery.AsyncResult(job_id)
        task_info = {
            "job_id": task.id,
            "status": task.status,
            "result": task.result if task.ready() else None,
        }

        # Check MongoDB
        results_collection = mongodb_database.get_collection("results")
        stored_result = await results_collection.find_one({"job_id": job_id})

        return {"task_info": task_info, "stored_result": stored_result is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}") from e

@app.get("/parameters/{algorithm_name}")
async def get_default_algorithm_parameters(
    algorithm_name: str = Path(description="The clustering algorithm to check for default parameters", default=..., examples=algorithms)
    ) -> dict[str, Any]:
    """
    Get the default parameters for a specific clustering algorithm.
    """
    algorithm = ALGORITHM_MAP.get(algorithm_name)
    if not algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {"clustering_params": algorithm.get_default_params()}

@app.get("/algorithms/")
async def get_available_algorithms() -> list[str]:
    """
    Get a list of available clustering algorithms.
    """
    return algorithms